# DSPH
# paper [Deep Semantic-aware Proxy Hashing for Multi-label Cross-modal Retrieval, TCSVT 2023]
# (https://ieeexplore.ieee.org/document/10149001)
import numpy as np
from tqdm import tqdm

import os
import torch

from torch.utils.data import DataLoader
import scipy.io as scio
import torch.nn

# from model.Evidencenet import EvidenceNet
# from model.TDML import MTDML, Psycho

from CLIP.train.FTH_crossmodal.hash_model_a import DHaPH, DHaPH_FCTH
# from model.modelbase import Baseclip
from CLIP.train.base import TrainBase
from CLIP.model.base.optimization import BertAdam
# from utils.get_args import get_args

import time
import torch.nn.functional as F

# from utils.LOSS import edl_log_loss
# from utils.MSLOSS_mi import MultiSimilarityLoss1
from .utils._utils import calc_learnable_params
# from get_args import get_args

from .utils.calc_utils import calc_map_k_matrix as calc_map_k, calc_map_k_from_dist, calc_recall_at_1
# from dataset.dataloader import dataloader, dataloader_open_set
from CLIP.dataset_zs.dataloader import dataloader
# from utils.intra import Estimate_Covariance, lamda_epoch, intra_synthetsis
# from losses_IAA_ms import ContrastiveLoss, MultiSimilarityLoss
# from utils.MSLOSS_mi import MultiSimilarityLoss
from .MSLoss import MSLoss
# from utils.Losses_s import Trip_hinge_loss, Trip_hinge_loss_Cross, ProxyNCA_Cross, ProxyAnchorLoss_Cross, TripletLoss2, \
#     TripletLoss3

from .IDMLloss import IDMLLoss, Tripletloss_ML

from pynvml import *

from .utils.sampler import ParametricSampler
from .get_args import get_args

class Trainer(TrainBase):

    def __init__(self, args, rank=0):
        args = get_args(args)
        args.rank = rank
        super(Trainer, self).__init__(args)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.cache = None
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        # HashModel = DHaPH
        HashModel = DHaPH_FCTH

        self.model = HashModel(
            outputDim=self.args.output_dim,
            clipPath=self.args.clip_path,
            # writer=self.writer,
            logger=self.logger,
            is_train=self.args.is_train,
            K=4,
            hard_routing=True,
            distill_weight=0.5
        ).to(self.rank).float()

        self.sampler1 = ParametricSampler(self.args)

        # if self.args.loss == 'Contrastive':
        #     self.criterion = ContrastiveLoss(batch_size=self.args.batch_size).to(self.rank)
        # elif self.args.loss == 'MS':
        if self.args.loss == 'MS':
            # self.criterion = TripletLoss3(opt=self.args).to(0)
            # self.criterion = IDMLLoss(gamma=self.args.gamma1, tau=self.args.tau1, square=False, normalized=False)
            self.criterion = Tripletloss_ML(square=False, normalized=False,
                                          sampler=self.sampler1).to(self.rank)

            self.criterion1 = MSLoss(temperature=self.args.tau, totalepoch=self.args.epochs, self_paced=True).to(self.rank)


        else:
            raise Exception('loss function not found!')

        self.logger.info(f"number of learnable params: {calc_learnable_params(self.model, self.criterion)}")

        to_opt = [
            {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},

            {'params': self.model.image_hash_com.parameters(), 'lr': self.args.lr},
            {'params': self.model.image_hash_fac.parameters(), 'lr': self.args.lr},

            {'params': self.model.text_hash_com.parameters(), 'lr': self.args.lr},
            {'params': self.model.text_hash_fac.parameters(), 'lr': self.args.lr},

            {'params': self.criterion.parameters(), 'lr': 0.015}
        ]

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))

        self.optimizer = BertAdam(to_opt, lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
                                  b1=0.9, b2=0.98, e=1e-6, t_total=5000 * self.args.epochs,
                                  weight_decay=1e-4, max_grad_norm=1.0)

        self.total_time = 0
        print(self.model)



    #
    # def _init_dataset(self):
    #     self.logger.info("init dataset.")
    #     self.logger.info(f"Using {self.args.dataset} dataset.")
    #     self.args.index_file = os.path.join("./dataset_zs/zs_nus17"
    #                                         , self.args.index_file)
    #     self.args.caption_file = os.path.join("./dataset_zs/zs_nus17", self.args.caption_file)
    #     self.args.label_file = os.path.join("./dataset_zs/zs_nus17", self.args.label_file)
    #     train_data, query_data, retrieval_data = dataloader(captionFile=self.args.caption_file,
    #                                                         indexFile=self.args.index_file,
    #                                                         labelFile=self.args.label_file,
    #                                                         maxWords=self.args.max_words,
    #                                                         imageResolution=self.args.resolution,
    #                                                         query_num=self.args.query_num,
    #                                                         train_num=self.args.train_num,
    #                                                         seed=self.args.seed)
    #     self.train_labels = train_data.get_all_label().to(0)
    #     self.query_labels = query_data.get_all_label()
    #     self.retrieval_labels = retrieval_data.get_all_label()
    #     self.args.retrieval_num = len(self.retrieval_labels)
    #     self.logger.info(f"query shape: {self.query_labels.shape}")
    #     self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
    #     self.train_loader = DataLoader(
    #         dataset=train_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=self.args.num_workers,
    #         pin_memory=True,
    #         shuffle=True
    #     )
    #     self.query_loader = DataLoader(
    #         dataset=query_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=self.args.num_workers,
    #         pin_memory=True,
    #         shuffle=True
    #     )
    #     self.retrieval_loader = DataLoader(
    #         dataset=retrieval_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=self.args.num_workers,
    #         pin_memory=True,
    #         shuffle=True
    #     )
    #
    # def _init_dataset(self):
    #     self.logger.info("init dataset.")
    #     if self.args.zero_shot:
    #         if self.args.train_dataset is None or self.args.eval_dataset is None:
    #             raise ValueError("Zero-shot mode requires --train-dataset and --eval-dataset")
    #
    #         train_name = self.args.train_dataset
    #         eval_name = self.args.eval_dataset
    #
    #         self.logger.info("==== ZERO-SHOT MODE ====")
    #     else:
    #         train_name = self.args.dataset
    #         eval_name = self.args.dataset
    #
    #         self.logger.info("==== NORMAL MODE ====")
    #
    #     self.logger.info(f"Train dataset: {train_name}")
    #     self.logger.info(f"Eval  dataset: {eval_name}")
    #
    #     train_dir, self.args.nclass = self._get_dataset_info(train_name)
    #
    #     train_index = os.path.join(train_dir, "index.mat")
    #     train_label = os.path.join(train_dir, "label.mat")
    #
    #     if 'nuswide' in train_name:
    #         train_caption = os.path.join(train_dir, "caption.txt")
    #     else:
    #         train_caption = os.path.join(train_dir, "caption.mat")
    #
    #     train_data, _, _ = dataloader(
    #         captionFile=train_caption,
    #         indexFile=train_index,
    #         labelFile=train_label,
    #         maxWords=self.args.max_words,
    #         imageResolution=self.args.resolution,
    #         query_num=self.args.query_num,
    #         train_num=self.args.train_num,
    #         seed=self.args.seed
    #     )
    #
    #
    #     eval_dir, _ = self._get_dataset_info(eval_name)
    #
    #     eval_index = os.path.join(eval_dir, "index.mat")
    #     eval_label = os.path.join(eval_dir, "label.mat")
    #
    #     if eval_name == 'nuswide':
    #         eval_caption = os.path.join(eval_dir, "caption.txt")
    #     else:
    #         eval_caption = os.path.join(eval_dir, "caption.mat")
    #
    #     _, query_data, retrieval_data = dataloader(
    #         captionFile=eval_caption,
    #         indexFile=eval_index,
    #         labelFile=eval_label,
    #         maxWords=self.args.max_words,
    #         imageResolution=self.args.resolution,
    #         query_num=self.args.query_num,
    #         train_num=self.args.train_num,
    #         seed=self.args.seed
    #     )
    #
    #     print(self.rank)
    #
    #     self.train_labels = train_data.get_all_label().to(self.rank)
    #     self.query_labels = query_data.get_all_label()
    #     self.retrieval_labels = retrieval_data.get_all_label()
    #     self.retrieval_num = len(self.retrieval_labels)
    #     self.logger.info(f"query shape: {self.query_labels.shape}")
    #     self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
    #
    #     self.train_loader = DataLoader(
    #         dataset=train_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=self.args.num_workers,
    #         pin_memory=True,
    #         shuffle=True
    #     )
    #     self.query_loader = DataLoader(
    #         dataset=query_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=self.args.num_workers,
    #         pin_memory=True,
    #         shuffle=True
    #     )
    #     self.retrieval_loader = DataLoader(
    #         dataset=retrieval_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=self.args.num_workers,
    #         pin_memory=True,
    #         shuffle=True
    #     )
    #
    #     self.max_loader = DataLoader(
    #         dataset=train_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=self.args.num_workers,
    #         pin_memory=True,
    #         shuffle=True
    #     )
    #
    #     self.n_samples = self.train_loader.dataset.__len__()



    # # 修改原有的_init_dataset方法
    # def _init_dataset(self, fold_idx=0):
    #     """初始化开放集数据集 - 使用预定义的类别划分"""
    #     self.logger.info("init open set dataset.")
    #     self.logger.info(f"Using {self.args.dataset} dataset.")
    #     self.logger.info(f"Fold index: {fold_idx}")
    #
    #     self.args.index_file = os.path.join("./dataset", self.args.dataset, self.args.index_file)
    #     self.args.caption_file = os.path.join("./dataset", self.args.dataset, self.args.caption_file)
    #     self.args.label_file = os.path.join("./dataset", self.args.dataset, self.args.label_file)
    #
    #     train_data, query_data, retrieval_data, unseen_classes = dataloader_open_set(
    #         captionFile=self.args.caption_file,
    #         indexFile=self.args.index_file,
    #         labelFile=self.args.label_file,
    #         maxWords=self.args.max_words,
    #         imageResolution=self.args.resolution,
    #         seed=self.args.seed,
    #         npy=False,
    #         fold_idx=fold_idx,
    #         args=self.args,
    #         logger=self.logger
    #     )
    #
    #     self.train_labels = train_data.get_all_label()
    #     self.query_labels = query_data.get_all_label()
    #     self.retrieval_labels = retrieval_data.get_all_label()
    #
    #     # 保存未见类信息
    #     self.unseen_classes = unseen_classes
    #     # 计算可见类（0-79中除去未见类）
    #     all_classes = set(range(80))
    #     self.seen_classes = np.array(sorted(list(all_classes - set(unseen_classes))))
    #
    #     self.args.retrieval_num = len(self.retrieval_labels)
    #     self.logger.info(f"query shape: {self.query_labels.shape}")
    #     self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
    #     self.logger.info(f"Seen classes in this fold: {len(self.seen_classes)}")
    #     self.logger.info(f"Unseen classes in this fold: {len(self.unseen_classes)} - {self.unseen_classes}")
    #
    #     self.train_loader = DataLoader(
    #         dataset=train_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=self.args.num_workers,
    #         pin_memory=True,
    #         shuffle=True
    #     )
    #     self.query_loader = DataLoader(
    #         dataset=query_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=self.args.num_workers,
    #         pin_memory=True,
    #         shuffle=True
    #     )
    #     self.retrieval_loader = DataLoader(
    #         dataset=retrieval_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=self.args.num_workers,
    #         pin_memory=True,
    #         shuffle=True
    #     )

    def add_noise_to_labels(self, labels):
        labels = labels.cpu().numpy()  # 转换为 numpy
        num_samples, num_labels = labels.shape
        num_noise = int(num_samples * self.args.noise_rate)

        noise_indices = np.random.choice(num_samples, num_noise, replace=False)

        for i in noise_indices:
            ones_indices = np.where(labels[i, :] == 1)[0]
            zeros_indices = np.where(labels[i, :] == 0)[0]

            if len(ones_indices) > 0:
                j = np.random.choice(ones_indices)
                labels[i, j] = 0

            if len(zeros_indices) > 0:
                j = np.random.choice(zeros_indices)
                labels[i, j] = 1

        return torch.tensor(labels, dtype=torch.float32).to(self.rank)  # 转换回 Tensor 并放回 GPU

    def print_gpu_usage(device=0):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU内存占用: {info.used // 1024 ** 2}MB / {info.total // 1024 ** 2}MB")

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d" % (epoch, self.args.epochs))

        all_loss = 0

        for image, text, label, index in tqdm(self.train_loader):

            if image.shape[0] < self.args.batch_size:
                continue

            # 带噪声标签
            noise_label = self.add_noise_to_labels(label)

            start_time = time.time()
            self.global_step += 1
            image = image.to(self.rank, non_blocking=True).float()
            text = text.to(self.rank, non_blocking=True)
            noise_label = noise_label.to(self.rank, non_blocking=True).float()
            self.model = self.model.to(self.rank)

            outputs = self.model(image, text)

            img_hash = outputs["image_embed"]
            text_hash = outputs["text_embed"]

            img_hash_d = outputs["image_embed_d"]
            text_hash_d = outputs["text_embed_d"]

            distill_loss = outputs["distill_loss"]

            # 多标签转 float
            noise_label = noise_label.float()

            img_hash, text_hash = F.normalize(img_hash), F.normalize(text_hash)
            img_hash_d, text_hash_d = F.normalize(img_hash_d), F.normalize(text_hash_d)

            # =====================================================
            # 1. 完整分支：跨模态损失
            # =====================================================
            loss_i2t = self.criterion(
                img_hash,
                noise_label,
                text_hash
            )

            loss_t2i = self.criterion(
                text_hash,
                noise_label,
                img_hash
            )

            loss_i2t1 = self.criterion1(
                img_hash,
                noise_label,
                text_hash, epoch
            )

            loss_t2i1 = self.criterion1(
                text_hash,
                noise_label,
                img_hash, epoch
            )

            loss_cm_com = loss_i2t + loss_t2i + loss_i2t1 + loss_t2i1

            # =====================================================
            # 2. 因子化路由分支：跨模态损失
            # =====================================================
            loss_i2t_d = self.criterion(
                img_hash_d,
                noise_label,
                text_hash_d
            )

            loss_t2i_d = self.criterion(
                text_hash_d,
                noise_label,
                img_hash_d
            )

            loss_i2t_d1 = self.criterion1(
                img_hash_d,
                noise_label,
                text_hash_d, epoch
            )

            loss_t2i_d1 = self.criterion1(
                text_hash_d,
                noise_label,
                img_hash_d, epoch
            )

            loss_cm_fac = loss_i2t_d + loss_t2i_d + loss_i2t_d1 + loss_t2i_d1

            # =====================================================
            # 3. 完整分支：模态内结构保持损失
            # =====================================================
            loss_i2i = self.criterion1(
                img_hash,
                noise_label,
                img_hash
            )

            loss_t2t = self.criterion1(
                text_hash,
                noise_label,
                text_hash
            )

            loss_intra_com = loss_i2i + loss_t2t

            # =====================================================
            # 4. 因子化路由分支：模态内结构保持损失
            # =====================================================
            loss_i2i_d = self.criterion1(
                img_hash_d,
                noise_label,
                img_hash_d
            )

            loss_t2t_d = self.criterion1(
                text_hash_d,
                noise_label,
                text_hash_d
            )

            loss_intra_fac = loss_i2i_d + loss_t2t_d

            # =====================================================
            # 5. 总损失
            # =====================================================
            lambda_fac = 0.5
            lambda_distill = 0.5
            # lambda_distill = 0.2
            # lambda_fac = 0
            # lambda_intra_com = 0
            # lambda_intra_fac = 0

            lambda_intra_com = 1.0
            lambda_intra_fac = 0.5

            loss = (
                    loss_cm_com
                    + lambda_fac * loss_cm_fac
                    + lambda_intra_com * loss_intra_com
                    + lambda_intra_fac * loss_intra_fac
                    + lambda_distill * distill_loss
            )

            all_loss += loss.data
            self.optimizer.zero_grad()
            # self.optimizerEvi.zero_grad()
            if loss != 0:
                loss.backward()

            self.optimizer.step()
            # self.optimizerEvi.step()
            self.total_time += time.time() - start_time
        # print(f"训练时间为：{time.time()-start_time1}")
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f' % itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}, time: {self.total_time}")
        # 在val()里面，算完mAP之后加上这句
        # self.measure_single_sample_inference()
        data_path, num_class = self._get_dataset_info(self.args.train_dataset)
        self.logger.info(f"train dataset:{self.args.train_dataset}, num_class={num_class}")
        data_path_eval, num_class_eval = self._get_dataset_info(self.args.eval_dataset)
        self.logger.info(f"eval dataset:{self.args.eval_dataset}, num_class_eval={num_class_eval}")

    def train(self):
        self.logger.info("Start train.")
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            if epoch %5 ==0:
                self.valid(epoch)

        self.logger.info(f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")

    # def get_code(self, data_loader, length: int):
    #
    #     img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
    #     text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
    #     encoder_time = 0
    #     for image, text, label, index in tqdm(data_loader):
    #         start_encoder_time = time.time()
    #         image = image.to(self.rank, non_blocking=True)
    #         text = text.to(self.rank, non_blocking=True)
    #         index = index.numpy()
    #         image_hash = self.model.encode_image(image)
    #         image_hash = torch.sign(image_hash)
    #         text_hash = self.model.encode_text(text)
    #         text_hash = torch.sign(text_hash)
    #         encoder_time = time.time() - start_encoder_time
    #
    #         img_buffer[index, :] = image_hash.data
    #         text_buffer[index, :] = text_hash.data
    #
    #     return img_buffer, text_buffer, encoder_time
    # def get_code(self, data_loader, length: int):
    #     img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
    #     text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
    #     total_encoder_time = 0.0
    #     total_samples = 0
    #
    #     # ========== GPU Warm-up （必须，和基线论文保持一致） ==========
    #     self.change_state(mode="valid")
    #     with torch.no_grad():
    #         for image, text, label, index in data_loader:
    #             image = image.to(self.rank, non_blocking=True)
    #             text = text.to(self.rank, non_blocking=True)
    #             _ = self.model.encode_image(image)
    #             _ = self.model.encode_text(text)
    #             break
    #     # ==============================================================
    #
    #     with torch.no_grad():
    #         for image, text, label, index in tqdm(data_loader):
    #             image = image.to(self.rank, non_blocking=True)
    #             text = text.to(self.rank, non_blocking=True)
    #             index = index.numpy()
    #             batch_size = image.shape[0]
    #
    #             torch.cuda.synchronize(device=self.rank)  # GPU同步，计时起点
    #             start_encoder_time = time.time()
    #
    #             image_hash = self.model.encode_image(image)
    #             image_hash = torch.sign(image_hash)
    #             text_hash = self.model.encode_text(text)
    #             text_hash = torch.sign(text_hash)
    #
    #             torch.cuda.synchronize(device=self.rank)  # GPU同步，计时终点
    #             batch_time = time.time() - start_encoder_time
    #
    #             total_encoder_time += batch_time
    #             total_samples += batch_size
    #
    #             img_buffer[index, :] = image_hash.data
    #             text_buffer[index, :] = text_hash.data
    #
    #     # 计算【单样本平均推理时间，单位ms】，对应表格里面的Inference Time
    #     avg_infer_ms = (total_encoder_time / total_samples) * 1000
    #     self.logger.info(
    #         f"[Inference Time Stat] total_samples: {total_samples}, total_time: {total_encoder_time:.4f}s, average per sample: {avg_infer_ms:.2f} ms")
    #     return img_buffer, text_buffer, avg_infer_ms

    # # 新增独立函数，专门测表格用的单样本推理时间 batch=1
    # def measure_single_sample_inference(self):
    #     self.change_state(mode="valid")
    #     import time
    #     with torch.no_grad():
    #         # warm-up
    #         dummy_img = torch.randn(1, 3, 224, 224).to(self.rank)
    #         dummy_txt = torch.randint(0, 49408, (1, 77)).to(self.rank)
    #         for _ in range(10):
    #             _img_hash = self.model.encode_image(dummy_img)
    #             _txt_hash = self.model.encode_text(dummy_txt)
    #
    #         # 正式计时
    #         test_num = 200
    #         total_time = 0.0
    #         for _ in range(test_num):
    #             img = torch.randn(1, 3, 224, 224).to(self.rank)
    #             txt = torch.randint(0, 49408, (1, 77)).to(self.rank)
    #             torch.cuda.synchronize(self.rank)
    #             s = time.time()
    #             h_img = self.model.encode_image(img)
    #             h_txt = self.model.encode_text(txt)
    #             torch.cuda.synchronize(self.rank)
    #             total_time += time.time() - s
    #         avg_ms = (total_time / test_num) * 1000
    #         self.logger.info(f"===== TABLE Inference Time (batch=1) =====")
    #         self.logger.info(f"Average per sample: {avg_ms:.2f} ms")

    # def valid(self, epoch):
    #     self.logger.info("Valid.")
    #     self.change_state(mode="valid")
    #
    #     # 获取训练集、查询集、图库特征
    #     query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
    #     # retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.args.retrieval_num)
    #     retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.retrieval_num)
    #
    #     mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
    #     mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
    #     mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
    #     mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
    #
    #     if self.max_mapi2t < mAPi2t:
    #         self.best_epoch_i = epoch
    #         self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t")
    #     self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
    #     if self.max_mapt2i < mAPt2i:
    #         self.best_epoch_t = epoch
    #         self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i")
    #     self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
    #     self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, \
    #                 MAX MAP(i->t): {self.max_mapi2t}, MAX MAP(t->i): {self.max_mapt2i}, query_encoder_time: {q_encoder_time}, retrieval_encoder_time: {r_encoder_time}")
    def valid(self, epoch):
        self.logger.info("Valid.")
        self.change_state(mode="valid")
        # 获取训练集、查询集、图库特征
        query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
        # retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.args.retrieval_num)
        retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.retrieval_num)
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)

        # ----------------------新增保存pth权重----------------------
        import os
        os.makedirs(self.args.save_dir, exist_ok=True)
        # i2t最优，保存模型权重
        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t")
            save_path_i2t = os.path.join(self.args.save_dir, "best_model_i2t.pth")
            torch.save(self.model.state_dict(), save_path_i2t)
            self.logger.info(f"save best i2t model pth to {save_path_i2t}")
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)

        # t2i最优，保存模型权重
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i")
            save_path_t2i = os.path.join(self.args.save_dir, "best_model_t2i.pth")
            torch.save(self.model.state_dict(), save_path_t2i)
            self.logger.info(f"save best t2i model pth to {save_path_t2i}")
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        # ----------------------------------------------------------

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, \
                    MAX MAP(i->t): {self.max_mapi2t}, MAX MAP(t->i): {self.max_mapt2i}, query_encoder_time: {q_encoder_time}, retrieval_encoder_time: {r_encoder_time}")

    # def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t"):
    #
    #     save_dir = os.path.join(self.args.save_dir, "PR_cruve")
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     query_img = query_img.cpu().detach().numpy()
    #     query_txt = query_txt.cpu().detach().numpy()
    #     retrieval_img = retrieval_img.cpu().detach().numpy()
    #     retrieval_txt = retrieval_txt.cpu().detach().numpy()
    #     query_labels = self.query_labels.numpy()
    #     retrieval_labels = self.retrieval_labels.numpy()
    #
    #     result_dict = {
    #         'q_img': query_img,
    #         'q_txt': query_txt,
    #         'r_img': retrieval_img,
    #         'r_txt': retrieval_txt,
    #         'q_l': query_labels,
    #         'r_l': retrieval_labels
    #     }
    #     scio.savemat(os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
    #     self.logger.info(f">>>>>> save best {mode_name} data!")