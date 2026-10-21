
import numpy as np
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
import scipy.io as scio
import torch.nn
from CLIP.train.FTH_crossmodal.hash_model_a import DHaPH, DHaPH_FCTH
from CLIP.train.base import TrainBase
from CLIP.model.base.optimization import BertAdam

import time
import torch.nn.functional as F
from .utils._utils import calc_learnable_params
from .utils.calc_utils import calc_map_k_matrix as calc_map_k, calc_map_k_from_dist, calc_recall_at_1
from CLIP.dataset_zs.dataloader import dataloader
from .MSLoss import MSLoss
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

        if self.args.loss == 'MS':
           
            ```

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

           ```
            start_time = time.time()
            self.global_step += 1
           
           ```
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

   
