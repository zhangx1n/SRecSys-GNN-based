#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author : Zhang Xin
@Contact: xinzhang_hp@163.com
@Time : 2022/4/13
"""
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torchmetrics.retrieval.hit_rate import RetrievalHitRate
from torchmetrics.retrieval.reciprocal_rank import RetrievalMRR
from torchmetrics.retrieval.recall import RetrievalRecall


class SRGNN(LightningModule):

    def __init__(self, lr, l2, lr_dc_step, lr_dc, model: torch.nn.Module, top_k=20):
        super(SRGNN, self).__init__()
        self.model = model
        self.top_k = top_k
        self.lr = lr
        self.l2 = l2
        self.lr_dc_step = lr_dc_step
        self.lr_dc = lr_dc
        self.loss_function = nn.CrossEntropyLoss()
        self.matric_hist = {
            "train/loss": [],
            "train/hit": [],
            "train/mrr": [],
            "test/loss": [],
            "test/hit": [],
            "test/mrr": []
        }
        self.train_hit = RetrievalHitRate()
        self.test_hit = RetrievalHitRate()
        self.train_mrr = RetrievalMRR()
        self.test_mrr = RetrievalMRR()
        self.train_recall = RetrievalRecall()
        self.test_recall = RetrievalRecall()

    def forward(self, data):
        return self.model(data)

    def step(self, batch: Any):
        scores = self.model(batch)
        targets = batch.y - 1
        loss = self.loss_function(scores, targets)
        return loss, scores, targets

    def training_step(self, batch, batch_id):
        loss, scores, targets = self.step(batch)
        sub_scores = scores.topk(self.top_k)[1]  # batch * top_k
        for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
            self.matric_hist["train/hit"].append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                self.matric_hist["train/mrr"].append(0)
            else:
                self.matric_hist["train/mrr"].append(1 / (np.where(score == target)[0][0] + 1))
        hit = np.mean(self.matric_hist["train/hit"]) * 100
        mrr = np.mean(self.matric_hist["train/mrr"]) * 100
        # print("scores: ", scores)
        # print("targets: ", targets)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/hit_rate", hit, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train/mrr", mrr, prog_bar=False, on_step=True, on_epoch=False)
        return {"loss": loss, "scores": scores, "targets": targets}

    # def training_epoch_end(self, outputs: List[Any]) -> None:
    #     hit, mrr = self.calculate_metric(outputs)
        # hit, mrr = 0, 0
        # self.log("train/hit", hit, prog_bar=True)
        # self.log("train/mrr", mrr, prog_bar=True)
        # self.log("train/hit_best", max(self.matric_hist["train/hit"]), prog_bar=False)

    # def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
    #     loss, scores, targets = outputs
    #     hit_rate = self.train_hit(scores, targets, self.top_k)

    def test_step(self, batch, batch_idx):
        loss, scores, targets = self.step(batch)
        sub_scores = scores.topk(self.top_k)[1]  # batch * top_k
        for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
            self.matric_hist["train/hit"].append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                self.matric_hist["train/mrr"].append(0)
            else:
                self.matric_hist["train/mrr"].append(1 / (np.where(score == target)[0][0] + 1))
        hit = np.mean(self.matric_hist["train/hit"]) * 100
        mrr = np.mean(self.matric_hist["train/mrr"]) * 100
        # print("scores: ", scores)
        # print("targets: ", targets)
        self.log("test/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("test/hit_rate", hit, prog_bar=False, on_step=True, on_epoch=False)
        self.log("test/mrr", mrr, prog_bar=False, on_step=True, on_epoch=False)
        return {"loss": loss, "scores": scores, "targets": targets}

    # def test_epoch_end(self, outputs: List[Any]):
    #     # hit, mrr = self.calculate_metric(outputs)
    #     hit, mrr = 0, 0
    #     self.log("test/hit", hit, prog_bar=True)
    #     self.log("test/mrr", mrr, prog_bar=True)
    #     self.log("train/hit_best", max(self.matric_hist["train/hit"]), prog_bar=True)

    # def on_epoch_end(self):
    #     # reset metrics at the end of every epoch
    #     self.matric_hist = {
    #         "train/loss": [],
    #         "train/hit": [],
    #         "train/mrr": [],
    #         "test/loss": [],
    #         "test/hit": [],
    #         "test/mrr": []
    #     }
    #

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_dc_step, gamma=self.lr_dc)
        return [optimizer], [scheduler]

    # def calculate_metric(self, outputs: List[Any]):
    #     # loss = torch.cat([x["loss"] for x in outputs], dim=0)
    #     scores = torch.cat([x["scores"] for x in outputs], dim=0)
    #     targets = torch.cat([x["targets"] for x in outputs], dim=0)
    #
    #     sub_scores = scores.topk(self.top_k)[1]  # batch * top_k
    #     for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
    #         self.matric_hist["train/hit"].append(np.isin(target, score))
    #         if len(np.where(score == target)[0]) == 0:
    #             self.matric_hist["train/mrr"].append(0)
    #         else:
    #             self.matric_hist["train/mrr"].append(1 / (np.where(score == target)[0][0] + 1))
    #     hit = np.mean(self.matric_hist["train/hit"]) * 100
    #     mrr = np.mean(self.matric_hist["train/mrr"]) * 100
    #     return hit, mrr
