import faiss
import torch
import numpy as np
import torch.nn.functional as F

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_map_k_matrix_u(qB, rB, query_L, retrieval_L, k=None, rank=0,
                        qB_u=None, rB_u=None, gamma=0.0, tau=5.0, square=True):
    """
    使用内省式相似度计算mAP（分块计算，避免显存溢出）
    """
    num_query = query_L.shape[0]
    num_retrieval = retrieval_L.shape[0]
    device = qB.device

    if k is None:
        k = num_retrieval

    # 归一化特征（提前归一化，避免重复计算）
    S_query = F.normalize(qB, dim=1)
    S_retrieval = F.normalize(rB, dim=1)

    # 处理不确定性特征
    if qB_u is None or qB_u.shape[1] != qB.shape[1]:
        U_query = torch.zeros_like(qB)
    else:
        U_query = F.normalize(qB_u, dim=1)

    if rB_u is None or rB_u.shape[1] != rB.shape[1]:
        U_retrieval = torch.zeros_like(rB)
    else:
        U_retrieval = F.normalize(rB_u, dim=1)

    # 计算相关性矩阵（整个计算，这个通常较小）
    # 确保在CPU上计算相关性矩阵，避免显存占用
    query_L_cpu = query_L.cpu()
    retrieval_L_cpu = retrieval_L.cpu()
    gnd = (query_L_cpu @ retrieval_L_cpu.t() > 0).float()  # [num_query, num_retrieval] 在CPU上

    # 设置分块大小（根据显存调整）
    batch_size = min(256, num_query)  # 每批处理的query数量
    total_map = 0.0
    valid_queries = 0

    for i in range(0, num_query, batch_size):
        end_idx = min(i + batch_size, num_query)
        batch_size_actual = end_idx - i

        # 获取当前batch的数据（保持在GPU上）
        S_query_batch = S_query[i:end_idx]  # [batch, bits]
        U_query_batch = U_query[i:end_idx]  # [batch, bits]

        # 获取当前batch的相关性标签（从CPU的gnd中取）
        gnd_batch = gnd[i:end_idx]  # [batch, num_retrieval] 在CPU上

        # 计算余弦相似度（GPU上）
        cos_sim_batch = S_query_batch @ S_retrieval.T  # [batch, num_retrieval]

        # 计算alpha: norm(s1 - s2) - 使用优化公式
        alpha = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_sim_batch, min=1e-8))
        if square:
            alpha = alpha ** 2

        # 计算beta: norm(u1 + u2)
        u_norm_sq = (U_query_batch ** 2).sum(dim=1, keepdim=True)
        v_norm_sq = (U_retrieval ** 2).sum(dim=1, keepdim=True).T
        uv_dot = U_query_batch @ U_retrieval.T
        beta_sq = u_norm_sq + v_norm_sq + 2.0 * uv_dot
        beta = torch.sqrt(torch.clamp(beta_sq, min=1e-8))
        if square:
            beta = beta_sq

        # 计算内省式相似度（GPU上）
        beta_r = (beta + gamma) / (alpha + 1e-8)
        S_in = 1.0 - (1.0 - cos_sim_batch) * torch.exp(-beta_r / tau)

        # 将S_in移到CPU进行计算（避免GPU内存压力）
        S_in_cpu = S_in.cpu()

        # 计算batch内每个query的AP（在CPU上计算）
        for j in range(batch_size_actual):
            gnd_j = gnd_batch[j]  # [num_retrieval] 在CPU上
            sim_j = S_in_cpu[j]  # [num_retrieval] 在CPU上

            # 按相似度降序排序（CPU上）
            _, idx = torch.sort(sim_j, descending=True)
            idx = idx[:k]

            # 确保idx和gnd_j都在CPU上
            idx = idx.cpu() if hasattr(idx, 'cpu') else idx
            gnd_sorted = gnd_j[idx]

            # 计算相关样本总数
            total_relevant = gnd_j.sum().item()
            if total_relevant == 0:
                continue

            # 计算平均精度
            relevant_count = 0
            precision_sum = 0.0

            for pos, rel in enumerate(gnd_sorted):
                if rel == 1:
                    relevant_count += 1
                    precision_sum += relevant_count / (pos + 1)

            map_j = precision_sum / min(total_relevant, k)
            total_map += map_j
            valid_queries += 1

        # 清理中间变量
        del S_query_batch, U_query_batch, gnd_batch, cos_sim_batch
        del alpha, beta, beta_r, S_in, S_in_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    map = total_map / valid_queries if valid_queries > 0 else 0
    return map


def calc_map_k_matrix_u_chunked_gnd(qB, rB, query_L, retrieval_L, k=None, rank=0,
                                    qB_u=None, rB_u=None, gamma=0.0, tau=5.0, square=True):
    """
    当相关性矩阵也过大时的分块计算版本
    """
    num_query = query_L.shape[0]
    num_retrieval = retrieval_L.shape[0]
    device = qB.device

    if k is None:
        k = num_retrieval

    # 归一化特征
    S_query = F.normalize(qB, dim=1)
    S_retrieval = F.normalize(rB, dim=1)

    # 处理不确定性特征
    if qB_u is None or qB_u.shape[1] != qB.shape[1]:
        U_query = torch.zeros_like(qB)
    else:
        U_query = F.normalize(qB_u, dim=1)

    if rB_u is None or rB_u.shape[1] != rB.shape[1]:
        U_retrieval = torch.zeros_like(rB)
    else:
        U_retrieval = F.normalize(rB_u, dim=1)

    # 分块大小
    query_batch_size = min(256, num_query)
    retrieval_batch_size = min(5000, num_retrieval)  # 检索集也分块

    total_map = 0.0
    valid_queries = 0

    for i in range(0, num_query, query_batch_size):
        q_end = min(i + query_batch_size, num_query)
        q_batch = slice(i, q_end)

        S_query_batch = S_query[q_batch]
        U_query_batch = U_query[q_batch]

        # 存储当前batch的AP
        batch_aps = []

        # 对检索集也分块计算相似度
        for j in range(0, num_retrieval, retrieval_batch_size):
            r_end = min(j + retrieval_batch_size, num_retrieval)
            r_batch = slice(j, r_end)

            S_retrieval_batch = S_retrieval[r_batch]
            U_retrieval_batch = U_retrieval[r_batch]

            # 计算余弦相似度
            cos_sim_batch = S_query_batch @ S_retrieval_batch.T

            # 计算alpha
            alpha = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_sim_batch, min=1e-8))
            if square:
                alpha = alpha ** 2

            # 计算beta
            u_norm_sq = (U_query_batch ** 2).sum(dim=1, keepdim=True)
            v_norm_sq = (U_retrieval_batch ** 2).sum(dim=1, keepdim=True).T
            uv_dot = U_query_batch @ U_retrieval_batch.T
            beta_sq = u_norm_sq + v_norm_sq + 2.0 * uv_dot
            beta = torch.sqrt(torch.clamp(beta_sq, min=1e-8))
            if square:
                beta = beta_sq

            # 内省式相似度
            beta_r = (beta + gamma) / (alpha + 1e-8)
            S_in = 1.0 - (1.0 - cos_sim_batch) * torch.exp(-beta_r / tau)

            # 计算当前检索块的相关性
            query_L_batch = query_L[q_batch]  # [batch, num_class]
            retrieval_L_batch = retrieval_L[r_batch]  # [retrieval_batch, num_class]
            gnd_batch = (query_L_batch @ retrieval_L_batch.t() > 0).float()

            # 存储当前块的相似度和标签
            for batch_idx in range(S_in.shape[0]):
                if len(batch_aps) <= batch_idx:
                    batch_aps.append({'sim': [], 'gnd': []})
                batch_aps[batch_idx]['sim'].append(S_in[batch_idx])
                batch_aps[batch_idx]['gnd'].append(gnd_batch[batch_idx])

            del cos_sim_batch, alpha, beta, beta_r, S_in, gnd_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 合并每个query的检索结果并计算AP
        for batch_idx in range(len(batch_aps)):
            # 合并所有检索块的相似度和标签
            all_sim = torch.cat(batch_aps[batch_idx]['sim'], dim=0)
            all_gnd = torch.cat(batch_aps[batch_idx]['gnd'], dim=0)

            # 排序
            _, idx = torch.sort(all_sim, descending=True)
            idx = idx[:k]
            gnd_sorted = all_gnd[idx]

            total_relevant = all_gnd.sum().item()
            if total_relevant == 0:
                continue

            relevant_count = 0
            precision_sum = 0.0
            for pos, rel in enumerate(gnd_sorted):
                if rel == 1:
                    relevant_count += 1
                    precision_sum += relevant_count / (pos + 1)

            map_j = precision_sum / min(total_relevant, k)
            total_map += map_j
            valid_queries += 1

        del batch_aps, S_query_batch, U_query_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_map / valid_queries if valid_queries > 0 else 0

# def calc_map_k_matrix_u(qB, rB, query_L, retrieval_L, k=None, rank=0,
#                         qB_u=None, rB_u=None, gamma=0.0, tau=5.0, square=True):
#     """
#     使用内省式相似度计算mAP
#     """
#
#     num_query = query_L.shape[0]
#     device = qB.device
#
#     if k is None:
#         k = retrieval_L.shape[0]
#
#     # 归一化特征
#     S_query = F.normalize(qB, dim=1)
#     S_retrieval = F.normalize(rB, dim=1)
#
#     # 处理不确定性特征
#     if qB_u is None:
#         U_query = torch.zeros_like(qB)
#     else:
#         U_query = F.normalize(qB_u, dim=1) if qB_u.shape[1] == qB.shape[1] else torch.zeros_like(qB)
#
#     if rB_u is None:
#         U_retrieval = torch.zeros_like(rB)
#     else:
#         U_retrieval = F.normalize(rB_u, dim=1) if rB_u.shape[1] == rB.shape[1] else torch.zeros_like(rB)
#
#     # 计算所有query与所有retrieval的相似度矩阵
#     cos_sim = S_query @ S_retrieval.T  # [num_query, num_retrieval]
#
#     # 计算alpha: norm(s1 - s2)
#     S_query_exp = S_query.unsqueeze(1)  # [num_query, 1, bits]
#     S_retrieval_exp = S_retrieval.unsqueeze(0)  # [1, num_retrieval, bits]
#     alpha = torch.norm(S_query_exp - S_retrieval_exp, dim=2)  # [num_query, num_retrieval]
#     # alpha = 1
#
#     # 计算beta: norm(u1 + u2)
#     U_query_exp = U_query.unsqueeze(1)  # [num_query, 1, bits]
#     U_retrieval_exp = U_retrieval.unsqueeze(0)  # [1, num_retrieval, bits]
#     beta = torch.norm(U_query_exp + U_retrieval_exp, dim=2)  # [num_query, num_retrieval]
#
#     if square:
#         alpha = alpha ** 2
#         beta = beta ** 2
#
#     # 计算内省式相似度
#     beta_r = (beta + gamma) / (alpha + 1e-8)
#     S_in = 1 - (1 - cos_sim) * torch.exp(-beta_r / tau)  # [num_query, num_retrieval]
#
#     # 计算相关性矩阵
#     gnd = (query_L @ retrieval_L.t() > 0).float()  # [num_query, num_retrieval]
#
#     map = 0.0
#     for i in range(num_query):
#         # 获取当前query的相关性标签和相似度分数
#         gnd_i = gnd[i]  # [num_retrieval]
#         sim_i = S_in[i]  # [num_retrieval]
#
#         # 按相似度降序排序
#         _, idx = torch.sort(sim_i, descending=True)
#         idx = idx.to(gnd_i.device)
#
#         gnd_sorted = gnd_i[idx]
#
#         # 计算相关样本总数
#         total_relevant = gnd_i.sum().item()
#         if total_relevant == 0:
#             continue
#
#         # 计算平均精度
#         relevant_count = 0
#         precision_sum = 0.0
#
#         for j in range(min(k, len(gnd_sorted))):
#             if gnd_sorted[j] == 1:
#                 relevant_count += 1
#                 precision_sum += relevant_count / (j + 1)
#
#         if relevant_count > 0:
#             map += precision_sum / min(total_relevant, k)
#
#     map = map / num_query if num_query > 0 else 0
#     return map

def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def calc_map_k_matrix(qB, rB, query_L, retrieval_L, k=None, rank=0):
    
    num_query = query_L.shape[0]
    if qB.is_cuda:
        qB = qB.cpu()
        rB = rB.cpu()
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        gnd = (query_L[iter].unsqueeze(0).mm(retrieval_L.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_recall_at_1(qB, rB, query_L, retrieval_L, k=None, rank=0):
    """
    计算 Recall@1
    参数:
        qB: 查询样本的哈希码 [num_query, bit] (如图像)
        rB: 检索库的哈希码 [num_retrieval, bit] (如文本)
        query_L: 查询样本的标签 [num_query, num_classes]
        retrieval_L: 检索库的标签 [num_retrieval, num_classes]
    """
    # 确保数据在CPU上
    if qB.is_cuda:
        qB = qB.cpu()
        rB = rB.cpu()
        query_L = query_L.cpu()
        retrieval_L = retrieval_L.cpu()

    # 使用FAISS进行最近邻搜索
    faiss_search_index = faiss.IndexFlatL2(rB.shape[-1])
    faiss_search_index.add(rB.numpy())

    # 搜索每个查询样本的最近1个邻居
    _, k_nn = faiss_search_index.search(qB.numpy(), 1)

    # 获取最近邻的标签
    k_nn_labels = retrieval_L[k_nn[:, 0]]  # 形状: [num_query, num_classes]

    # 计算Recall@1
    recall_at_1 = 0
    for i in range(len(query_L)):
        query_label = query_L[i]
        retrieved_label = k_nn_labels[i]

        # 检查是否有共同的标签
        if ((query_label * retrieved_label).sum() > 0):
            recall_at_1 += 1

    recall_at_1 = recall_at_1 / len(query_L)
    return recall_at_1

def calc_map_k(qB, rB, query_L, retrieval_L, k=None, rank=0):

    num_query = query_L.shape[0]
    qB = torch.sign(qB)
    rB = torch.sign(rB)
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)      # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.to(rank)
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_map_k_from_dist(dist_matrix, qB, rB, query_L, retrieval_L, k=None, rank=0):
    num_query = query_L.shape[0]
    map = 0.0
    if k is None:
        k = retrieval_L.shape[0]

    # 确保输入为CPU Tensor
    dist_matrix = dist_matrix.cpu().float()
    retrieval_L = retrieval_L.cpu()
    query_L = query_L.cpu()

    for i in range(num_query):
        q_label = query_L[i].unsqueeze(0)  # 保持二维形状 [1, C]
        gnd = (q_label @ retrieval_L.T > 0).squeeze().float()  # [M]

        # 按距离升序排序（越小越相关）
        _, indices = torch.sort(dist_matrix[i])
        gnd_sorted = gnd[indices]

        # 计算相关样本数
        total_relevant = torch.sum(gnd_sorted[:k])
        if total_relevant == 0:
            continue

        # 计算平均精度
        relevant_pos = torch.nonzero(gnd_sorted[:k]).squeeze() + 1  # 位置从1开始
        precision_at_k = torch.arange(1, total_relevant + 1).float() / relevant_pos.float()
        map += torch.mean(precision_at_k)

    map /= num_query
    return map.item()

# #
# def calc_map_k_from_dist(dist_matrix, qB, rB, query_L, retrieval_L, k=None):
#     num_query = query_L.shape[0]
#     map = 0.0
#     if k is None:
#         k = retrieval_L.shape[0]
#
#     # 确保所有计算在CPU进行（评估指标不需要GPU加速）
#     dist_matrix = dist_matrix.cpu()  # 关键修改1：将距离矩阵移到CPU
#
#     for i in range(num_query):
#         q_label = query_L[i]
#         gnd = (q_label @ retrieval_L.T > 0).squeeze().float()  # 相关性标签
#
#         # 按距离排序（从小到大）
#         _, indices = torch.sort(dist_matrix[i])  # 此时 indices 在CPU上
#
#         gnd_sorted = gnd[indices]  # 关键修改2：确保 gnd 和 indices 都在CPU
#
#         # 计算 AP
#         total_relevant = torch.sum(gnd_sorted[:k])
#         if total_relevant == 0:
#             continue
#         relevant_pos = torch.nonzero(gnd_sorted[:k]).squeeze() + 1
#         precision_at_k = torch.arange(1, total_relevant + 1).float() / relevant_pos.float()
#         map += torch.mean(precision_at_k)
#
#     map /= num_query
#     return map.item()
# #

def calc_precisions_topn_matrix(qB, rB, query_L, retrieval_L, recall_gas=0.02, num_retrieval=10000):
    if not isinstance(qB, torch.Tensor):
        qB = torch.from_numpy(qB)
        rB = torch.from_numpy(rB)
        query_L = torch.from_numpy(query_L)
        retrieval_L = torch.from_numpy(retrieval_L)
    qB = qB.float()
    rB = rB.float()
    qB = torch.sign(qB - 0.5)
    rB = torch.sign(rB - 0.5)
    if qB.is_cuda:
        qB = qB.cpu()
        rB = rB.cpu()
    num_query = query_L.shape[0]
    # num_retrieval = retrieval_L.shape[0]
    precisions = [0] * int(1 / recall_gas)
    gnds = (query_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
    hamms = calc_hammingDist(qB, rB)
    _, inds = torch.sort(hamms, dim=-1)
    for iter in range(num_query):
        gnd = gnds[iter]
        ind = inds[iter]

        gnd = gnd[ind]
        for i, recall in enumerate(np.arange(recall_gas, 1 + recall_gas, recall_gas)):
            total = int(num_retrieval * recall)
            right = torch.nonzero(gnd[: total]).squeeze().numpy()

            right_num = right.size
            precisions[i] += (right_num/total)
    for i in range(len(precisions)):
        precisions[i] /= num_query
    return precisions


def calc_precisions_topn(qB, rB, query_L, retrieval_L, recall_gas=0.02, num_retrieval=10000):
    qB = qB.float()
    rB = rB.float()
    qB = torch.sign(qB - 0.5)
    rB = torch.sign(rB - 0.5)
    num_query = query_L.shape[0]
    # num_retrieval = retrieval_L.shape[0]
    precisions = [0] * int(1 / recall_gas)
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)  # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        for i, recall in enumerate(np.arange(recall_gas, 1 + recall_gas, recall_gas)):
            total = int(num_retrieval * recall)
            right = torch.nonzero(gnd[: total]).squeeze().numpy()
            # right_num = torch.nonzero(gnd[: total]).squeeze().shape[0]
            right_num = right.size
            precisions[i] += (right_num/total)
    for i in range(len(precisions)):
        precisions[i] /= num_query
    return precisions


def calc_precisions_hash(qB, rB, query_L, retrieval_L):
    qB = qB.float()
    rB = rB.float()
    qB = torch.sign(qB - 0.5)
    rB = torch.sign(rB - 0.5)
    num_query = query_L.shape[0]
    num_retrieval = retrieval_L.shape[0]
    bit = qB.shape[1]
    hamm = calc_hammingDist(qB, rB)
    hamm = hamm.type(torch.ByteTensor)
    total_num = [0] * (bit + 1)
    max_hamm = int(torch.max(hamm))
    gnd = (query_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze()
    total_right = torch.sum(torch.matmul(query_L, retrieval_L.t())>0)
    precisions = np.zeros([max_hamm + 1])
    recalls = np.zeros([max_hamm + 1])
    # _, index = torch.sort(hamm)
    # del _
    # for i in range(index.shape[0]):
    #     gnd[i, :] = gnd[i, index[i]]
    # del index
    right_num = 0
    recall_num = 0
    for i, radius in enumerate(range(0, max_hamm+1)):
        recall = torch.nonzero(hamm == radius)
        right = gnd[recall.split(1, dim=1)]
        recall_num += recall.shape[0]
        del recall
        right_num += torch.nonzero(right).shape[0]
        del right
        precisions[i] += (right_num / (recall_num + 1e-8))
        # recalls[i] += (recall_num / num_retrieval / num_query)
        recalls[i] += (recall_num / total_right)
    return precisions, recalls

def calc_precisions_hash_my(qB, rB, *, Gnd, num_query, num_retrieval):
    if not isinstance(qB, torch.Tensor):
        qB = torch.from_numpy(qB)
    if not isinstance(rB, torch.Tensor):
        rB = torch.from_numpy(rB)
    if not isinstance(Gnd, torch.Tensor):
        Gnd = torch.from_numpy(Gnd)

    def CalcHammingDist_np(B1, B2):
        q = B2.shape[1]
        distH = 0.5 * (q - np.dot(B1, B2.transpose()))
        return distH
    bit = qB.shape[1]
    # if isinstance(qB, np.ndarray):
    #     hamm = CalcHammingDist_np(qB, rB)
    # else:
    hamm = calc_hammingDist(qB, rB)
    hamm = hamm.type(torch.ByteTensor)
    total_num = [0] * (bit + 1)
    max_hamm = int(torch.max(hamm))

    gnd = Gnd

    total_right = torch.sum(gnd>0)
    precisions = np.zeros([max_hamm + 1])
    recalls = np.zeros([max_hamm + 1])

    right_num = 0
    recall_num = 0
    for i, radius in enumerate(range(0, max_hamm+1)):
        recall = torch.nonzero(hamm == radius)
        right = gnd[recall.split(1, dim=1)]
        recall_num += recall.shape[0]
        del recall
        right_num += torch.nonzero(right).shape[0]
        del right
        precisions[i] += (right_num / (recall_num + 1e-8))
        recalls[i] += (recall_num / num_retrieval / num_query)
        # recalls[i] += (recall_num / total_right)
    p = precisions.round(2)
    r = recalls.round(2)
    # return p, r

    precisions = []
    recalls = []

    precision_ = 0
    num = 1
    for i in range(len(r) - 1):
        if r[i] == r[i + 1]:
            precision_ += p[i]
            num += 1
        else:
            precision_ += p[i]
            precisions.append(precision_ / num)
            recalls.append(r[i])
            precision_ = 0
            num = 1
            
    return np.asarray(precisions).round(2), np.asarray(recalls)


def calc_precisions_hamming_radius(qB, rB, query_L, retrieval_L, hamming_gas=1):
    num_query = query_L.shape[0]
    bit = qB.shape[1]
    precisions = [0] * int(bit / hamming_gas)
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)  # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        for i, recall in enumerate(np.arange(1, bit+1, hamming_gas)):
            total = torch.nonzero(hamm <= recall).squeeze().shape[0]
            if total == 0:
                precisions[i] += 0
                continue
            right = torch.nonzero(gnd[: total]).squeeze().numpy()
            right_num = right.size

            precisions[i] += (right_num / total)
    for i in range(len(precisions)):
        precisions[i] /= num_query
    return precisions


def calc_neighbor(label1, label2):
    # calculate the similar matrix
    Sim = label1.matmul(label2.transpose(0, 1)) > 0
    return Sim.float()


def norm_max_min(x: torch.Tensor, dim=None):
    if dim is None:
        max = torch.max(x)
        min = torch.min(x)
    if dim is not None:
        max = torch.max(x, dim=dim)[0]
        min = torch.min(x, dim=dim)[0]
        if dim > 0:
            max = max.unsqueeze(len(x.shape) - 1)
            min = min.unsqueeze(len(x.shape) - 1)
    norm = (x - min) / (max - min)
    return norm


def norm_mean(x: torch.Tensor, dim=None):
    if dim is None:
        mean = torch.mean(x)
        std = torch.std(x)
    if dim is not None:
        mean = torch.mean(x, dim=dim)
        std = torch.std(x, dim=dim)
        if dim > 0:
            mean = mean.unsqueeze(len(x.shape) - 1)
            std = std.unsqueeze(len(x.shape) - 1)
    norm = (x - mean) / std
    return norm


def norm_abs_mean(x: torch.Tensor, dim=None):
    if dim is None:
        mean = torch.mean(x)
        std = torch.std(x)
    if dim is not None:
        mean = torch.mean(x, dim=dim)
        std = torch.std(x, dim=dim)
        if dim > 0:
            mean = mean.unsqueeze(len(x.shape) - 1)
            std = std.unsqueeze(len(x.shape) - 1)
    norm = torch.abs(x - mean) / std
    return norm


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def calc_IF(all_bow):
    word_num = torch.sum(all_bow, dim=0)
    total_num = torch.sum(word_num)
    IF = word_num / total_num
    return IF
