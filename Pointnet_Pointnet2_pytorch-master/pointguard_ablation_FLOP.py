"""
Ablation Study:
get FLOP for three models
"""
from data_utils.ModelNetDataLoader_clean_per_inj import ModelNetDataLoader_clean_per_inj
from defense_utils.generative_adversarial_network import perturbation_attack, weighted_dist_per, add_ADchannel
import torch.nn.functional as F
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import json
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, f1_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ptflops import get_model_complexity_info

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def main():
    #   pointnet_sem_seg
    # ==================================================
    model = importlib.import_module("pointnet_sem_seg").get_model(num_class=2)
    macs, params = get_model_complexity_info(
        model,
        (4, 16),   # input shape (C, N)
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True
    )

    print("pointnet_sem_seg")
    print('MACs:', macs)
    print('Params:', params)

    #   pointnet_pointguard_small
    # ==================================================
    model = importlib.import_module("pointnet_pointguard_small").get_model()
    macs, params = get_model_complexity_info(
        model,
        (4, 16),   # input shape (C, N)
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True
    )

    print("pointnet_pointguard_small")
    print('MACs:', macs)
    print('Params:', params)

    #   pointguard_discriminator
    # ==================================================
    model = importlib.import_module("pointguard_discriminator").get_model()
    macs, params = get_model_complexity_info(
        model,
        (5, 16),   # input shape (C, N)
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True
    )

    print("pointguard_discriminator")
    print('MACs:', macs)
    print('Params:', params)


if __name__ == '__main__':
    main()
