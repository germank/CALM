# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
np.random.seed(42) # different seeds will generate different segmentations
import tqdm
from pprint import pprint

# model definition
def get_loss(x, y):
    return random.random()
def train(loss):
    pass

# CALM tasks
import calm
use_cuda = False
MultiLingual_dev_10k = calm.CALM("data/news_dev", "char", switch_frequency=10000, nswitches=100, window_size=20, batch_size=10, cuda=use_cuda)
# MultiLingual_dev_100k = calm.CALM("data/news_dev", "char", switch_frequency=100000, nswitches=100, window_size=20, batch_size=10, cuda=use_cuda)
# MultiDomain_dev_10k = calm.CALM("data/domain_dev", "char", switch_frequency=10000, nswitches=100, window_size=20, batch_size=10, cuda=use_cuda)
# MultiDomain_dev_20k = calm.CALM("data/domain_dev", "char", switch_frequency=20000, nswitches=100, window_size=20, batch_size=10, cuda=use_cuda)

# Usage:

vocab = MultiLingual_dev_10k.get_vocabulary()  #to construct the models
calm_it = MultiLingual_dev_10k
results_fn = "loss_log.jsonl"
with open(results_fn, "w") as flog:
    for x, y in tqdm.tqdm(calm_it, desc="Keep CALM and Learn"):
        loss = get_loss(x, y)
        train(loss)
        calm.log_loss(flog, calm_it, loss)
pprint(calm.compute_metrics(results_fn))
