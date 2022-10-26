# -*- coding: utf-8 -*-

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_recall_fscore_support

    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }


    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }


    def ddie_compute_metrics(task_name, preds, labels, every_type=False):
        label_list = ('Mechanism', 'Effect', 'Advise', 'Int')
        p, r, f, s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[1, 2, 3, 4], average='micro')
        result = {
            "Precision": p,
            "Recall": r,
            "microF": f
        }
        # if every_type:
        #
        #     p1,r1,f1,s1 = precision_recall_fscore_support(y_pred=preds,y_true=labels,labels=[1],average='micro')
        #     p2,r2,f2,s2 = precision_recall_fscore_support(y_pred=preds,y_true=labels,labels=[2],average='micro')
        #     p3,r3,f3,s3 = precision_recall_fscore_support(y_pred=preds,y_true=labels,labels=[3],average='micro')
        #     p4,r4,f4,s4 = precision_recall_fscore_support(y_pred=preds,y_true=labels,labels=[4],average='micro')
        #
        #     result['mechanism_f'] = f1
        #     result['effect_f'] = f2
        #     result['advice_f'] = f3
        #     result['int_f'] = f4
        return result


    def pretraining_compute_metrics(task_name, preds, labels, every_type=False):
        acc = accuracy_score(y_pred=preds, y_true=labels)
        result = {
            "Accuracy": acc,
        }
        return result
