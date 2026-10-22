from argparse import Namespace
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

import torch.nn.functional as F

class DistanceVisualizer:
    def __init__(self):
        self.distance_history = defaultdict(list)
        self.iteration_count = 0

    def update(self, d_an, labels):
        """记录当前batch的距离数据"""
        if isinstance(d_an, torch.Tensor):
            d_an = d_an.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        self.distance_history[self.iteration_count].extend(d_an)
        self.iteration_count += 1

    # def plot(self, save_path="anchor_negative_distance.png"):
    #     """生成并保存可视化图表"""
    #     iterations = []
    #     distances = []
    #     counts = []
    #
    #     for iter_num, dists in self.distance_history.items():
    #         unique_dists, unique_counts = np.unique(dists, return_counts=True)
    #         iterations.extend([iter_num] * len(unique_dists))
    #         distances.extend(unique_dists)
    #         counts.extend(unique_counts)
    #
    #     max_count = max(counts) if counts else 1
    #     alphas = [0.1 + 0.9 * (c / max_count) for c in counts]
    #
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     scatter = ax.scatter(iterations, distances, c='blue', alpha=alphas, s=10)
    #
    #     # 创建颜色条并明确指定关联的Axes
    #     sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=max_count))
    #     sm.set_array([])
    #     cbar = fig.colorbar(sm, ax=ax)
    #     cbar.set_label('Density')
    #
    #     ax.set_xlabel('Training Iteration')
    #     ax.set_ylabel('Anchor-Negative Distance')
    #     ax.set_title('Sampling Distribution')
    #     ax.grid(True, alpha=0.3)
    #
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()
    #     print(f"save {save_path}")

    def plot(self, save_path="anchor_negative_distance.png"):
        """Generate and save visualization"""
        iterations = []
        distances = []
        counts = []

        for iter_num, dists in self.distance_history.items():
            unique_dists, unique_counts = np.unique(dists, return_counts=True)
            iterations.extend([iter_num] * len(unique_dists))
            distances.extend(unique_dists)
            counts.extend(unique_counts)

        fig, ax = plt.subplots(figsize=(12, 6))

        # 使用计数作为颜色值
        scatter = ax.scatter(iterations, distances, c=counts,
                             cmap='Blues', alpha=0.8, s=10,
                             norm=plt.Normalize(vmin=min(counts), vmax=max(counts)))

        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Density')

        ax.set_xlabel('Training Iteration')
        ax.set_ylabel('Anchor-Negative Distance')
        ax.set_title('Sampling Distribution')
        ax.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

class ParametricSampler(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.n_bins, self.bin_limit = args.n_bins, args.bin_limit

        self.support = np.linspace(self.bin_limit[0], self.bin_limit[1], self.n_bins)

        self.no_norm_distr = self.give_distr(args.policy_init_distr)
        self.distr = self.norm(self.no_norm_distr)

        self.merge_oobs = args.policy_merge_oobs
        self.include_pos = args.policy_include_pos
        self.include_same = args.policy_include_same
        self.margin = 0.2
        self.args = args

    def give_distr(self, name):
        if name == "random":
            distr_to_init = np.array([1.0] * (self.n_bins - 1))
        # gaussian_fixed、binomial_fixed、chi2_fixed
        elif name == "gaussian_fixed":
            distr_to_init = self.gaussian_fixed()

        elif name == "binomial_fixed":
            distr_to_init = self.binomial_fixed()

        elif name == "chi2_fixed":
            distr_to_init = self.chi2_fixed()

        elif name == "distance":
            distr_to_init = self.probmode(self.support, self.n_bins, upper_lim=0.5, mode="distance")
        elif name == "uniform_low_and_tight":
            distr_to_init = self.probmode(self.support, self.n_bins, mu=0.4, sig=0.1, mode="uniform")
        elif name == "uniform_lowish_and_tight":
            distr_to_init = self.probmode(self.support, self.n_bins, mu=0.6, sig=0.1, mode="uniform")
        elif name == "heavyside_low":
            distr_to_init = self.probmode(self.support, self.n_bins, mu=0.5, mode="heavyside")
        elif name == "uniform_low":
            distr_to_init = self.probmode(self.support, self.n_bins, mu=0.5, sig=0.2, mode="uniform")
        elif name == "uniform_avg":
            distr_to_init = self.probmode(self.support, self.n_bins, mu=1.0, sig=0.2, mode="uniform")
        elif name == "uniform_high":
            distr_to_init = self.probmode(self.support, self.n_bins, mu=1.5, sig=0.2, mode="uniform")
        elif name == "normal_low":
            distr_to_init = self.probmode(self.support, self.n_bins, mu=0.5, sig=0.05, mode="gaussian")
        elif name == "normal_avg":
            distr_to_init = self.probmode(self.support, self.n_bins, mu=1.1, sig=0.04, mode="gaussian")
        elif name == "normal_high":
            distr_to_init = self.probmode(self.support, self.n_bins, mu=1.6, sig=0.04, mode="gaussian")
        elif name == "semi_hard":  # 新增专门模式
            # 在margin附近集中概率
            mu = (self.bin_limit[0] + self.bin_limit[1]) / 2 + 0.3 * 0.2
            sig = 0.2 / 2
            return self.probmode(self.support, self.n_bins, mu=mu, sig=sig, mode="gaussian")
        else:
            raise Exception("Init. Distr. >> {} << not available!".format(name))
        return distr_to_init

    def bin_centers(self):
        return (self.support[:-1] + self.support[1:]) / 2.0

    def gaussian_fixed(self):
        centers = self.bin_centers()
        mu = (self.bin_limit[0] + self.bin_limit[1]) / 2.0
        sigma = 0.25
        prob = np.exp(-0.5 * ((centers - mu) / sigma) ** 2)
        prob = np.clip(prob, 1e-15, None)
        return prob / prob.sum()

    def binomial_fixed(self):
        n = self.n_bins - 2
        k = np.arange(self.n_bins - 1)
        p = 0.5

        # log form for numerical stability
        from math import lgamma
        log_prob = (
                np.array([lgamma(n + 1) - lgamma(int(ki) + 1) - lgamma(n - int(ki) + 1) for ki in k])
                + k * np.log(p + 1e-12)
                + (n - k) * np.log(1 - p + 1e-12)
        )
        prob = np.exp(log_prob - np.max(log_prob))
        prob = np.clip(prob, 1e-15, None)
        return prob / prob.sum()

    def chi2_fixed(self):
        centers = self.bin_centers()
        df = 3.0

        # normalize distances to a positive range for chi-square density
        x = (centers - self.bin_limit[0]) / (self.bin_limit[1] - self.bin_limit[0] + 1e-12)
        x = x * df * 2.0 + 1e-6

        from math import gamma
        coef = 1.0 / (2.0 ** (df / 2.0) * gamma(df / 2.0))
        prob = coef * (x ** (df / 2.0 - 1.0)) * np.exp(-x / 2.0)
        prob = np.clip(prob, 1e-15, None)
        return prob / prob.sum()

    def norm(self, distr):
        return distr / np.sum(distr)

    def pdist(self, A, y = None):
        prod = torch.mm(A, y.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min=0)
        return res.clamp(min=0).sqrt()

    def pdist1(self, A, B=None):
        """计算跨模态距离矩阵（A为图像特征，B为文本特征）"""
        if B is None:
            B = A  # 单模态情况
        prod = torch.mm(A, B.t())  # 内积矩阵
        norm_A = A.pow(2).sum(dim=1).view(-1, 1)  # ||a||^2
        norm_B = B.pow(2).sum(dim=1).view(1, -1)  # ||b||^2
        return (norm_A + norm_B - 2 * prod).clamp(min=0).sqrt()  # L2距离

    def update_sample_distr(self, distr_change):
        """
        distr_change: 连续调整因子，形状 [n_bins]，范围 [0.5, 2.0]
        """
        # 乘性调整
        self.no_norm_distr = np.clip(self.no_norm_distr * distr_change, 1e-25, 1e25)
        self.distr = self.norm(self.no_norm_distr)

    def forward(self, batch, labels, y = None):
        # 确保标签是NumPy数组格式
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        if y is None:
            y = batch

        # 初始化存储三元组（anchor, positive, negative）索引的列表
        positives, negatives, anchors = [], [], []

        # 计算输入特征batch的成对L2距离矩阵
        distances = self.pdist(batch.detach(), y.detach())

        # 将距离值映射到预设的bin区间
        p_assigns = np.sum((distances.cpu().numpy().reshape(-1) > self.support[1:-1].reshape(-1, 1)).T, axis=1).reshape(
            distances.shape
        )
        # 这个索引矩阵后续会用于从分布 self.distr 中获取采样概率。
        # p_assigns = np.sum((distances.cpu().numpy().reshape(-1) > self.distr[1:-1].reshape(-1, 1)).T, axis=1).reshape(
        #     distances.shape
        # )

        # # 在forward方法中会使用更新后的分布
        # 根据bin索引从分布self.distr中获取采样概率
        sample_ps = self.distr[p_assigns]

        # 将超出bin范围（Out-of-Bounds, OOB）的距离采样概率置零
        if not self.merge_oobs:
            outside_support_lim = (distances.cpu().numpy().reshape(-1) < self.bin_limit[0]) * (
                distances.cpu().numpy().reshape(-1) > self.bin_limit[1]
            )
            outside_support_lim = outside_support_lim.reshape(distances.shape)
            sample_ps[outside_support_lim] = 0

        bs = len(batch)
        # 计算标签相似性矩阵（N×N布尔矩阵）,样本对共享至少一个标签
        # 结果sim是[bs, bs]的布尔矩阵，sim[i,j]=True表示样本i和j是相似的(正样本对)
        sim = labels @ labels.T > 0
        # 三元组采样循环
        # 遍历批次中的每个样本，作为anchor(锚点)
        for i in range(bs):
            # sim[i]: 取出第i个样本与其他所有样本的相似性(布尔向量)
            # 取反得到负样本掩码(不相似的样本)
            neg = ~sim[i]
            # 正样本掩码(相似的样本)
            pos = sim[i]
            # 将当前样本索引i加入anchors列表
            anchors.append(i)
            # 将anchor自身的正样本标记设为False
            if not self.include_same:
                pos[i] = 0
            # 正样本采样: 随机选择一个正样本, 若没有正样本则跳过当前anchor
            pos_idxes = np.where(pos)[0]
            if len(pos_idxes) == 0:
                continue
            # 随机选择一个正样本索引，加入positives列表
            positives.append(np.random.choice(pos_idxes))

            # 负样本采样
            if self.include_pos:
                sample_p = sample_ps[i]
                sample_p = sample_p / sample_p.sum()
                negatives.append(np.random.choice(bs, p=sample_p))
            # 仅从真正的负样本中采样
            else:
                # 只考虑真正负样本的采样概率
                sample_p = sample_ps[i][neg]
                sample_p_sum = sample_p.sum()
                # 检查是否有有效的负样本(概率和不为0)
                if sample_p_sum == 0:
                    continue
                # 归一化负样本概率分布
                sample_p = sample_p / sample_p_sum
                # 从负样本中按概率采样一个样本
                negatives.append(np.random.choice(np.arange(bs)[neg], p=sample_p))

        # 将anchors、positives和negatives三个列表按位置组合成三元组列表
        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        # 返回采样得到的三元组
        return sampled_triplets

    # mu: 均值(默认1)  sig: 标准差(默认1)
    def gaussian(self, x, mu=1, sig=1):
        # 计算标准高斯分布的概率密度     在均值处概率密度最高，向两侧对称递减
        return 1 / np.sqrt(2 * np.pi * sig) * np.exp(-((x - mu) ** 2) / (2 * sig**2))

    # 创建区间内的均匀分布   mu: 区间中心(默认1)   sig: 区间半径(默认0.25)
    def uniform(self, x, mu=1, sig=0.25):
        # 创建输入数组的副本sp
        sp = x.copy()
        # 将区间[mu-sig, mu+sig]内的值设为1
        sp[(x >= mu - sig) * (x <= mu + sig)] = 1
        # 区间外的值设为0        在指定区间内概率均匀分布
        sp[(x < mu - sig) + (x > mu + sig)] = 0
        # 对特定距离范围内的样本进行均匀采样
        return sp

    # 创建基于距离阈值的阶跃分布, upper_lim: 距离阈值上限      简单的二分分布
    def distance(self, x, upper_lim):
        sp = x.copy()
        # 小于等于阈值的设为1
        sp[x <= upper_lim] = 1
        # 大于阈值的设为0
        sp[x >= upper_lim] = 0
        # 优先采样距离小于阈值的样本
        return sp

    def heavyside(self, x, mu):
        sp = x.copy()
        sp[x <= mu] = 1
        sp[x > mu] = 0
        return sp

    def probmode(self, support_space, support, mu=None, sig=None, upper_lim=None, mode="uniform"):
        if mode == "uniform":
            # support_space[:-1]: 去除最后一个边界点，得到每个bin的起始边界
            # + 2 / support: 将bin边界偏移半个bin宽度(调整到bin中心)
            probdist = self.uniform(support_space[:-1] + 2 / support, mu=mu, sig=sig)
        elif mode == "gaussian":
            probdist = self.gaussian(support_space[:-1] + 2 / support, mu=mu, sig=sig)
        elif mode == "distance":
            probdist = self.distance(support_space[:-1] + 2 / support, upper_lim)
        elif mode == "heavyside":
            probdist = self.heavyside(support_space[:-1] + 2 / support, mu=mu)
        # 将概率值裁剪到[1e-15, 1]范围内, 避免后续计算中出现数值问题(如除以零)
        probdist = np.clip(probdist, 1e-15, 1)
        # 概率归一化
        probdist = probdist / probdist.sum()
        return probdist

