"""Contains the PaccMann model."""

import logging
import sys
from collections import OrderedDict

import pytoda
import torch
import torch.nn as nn
from pytoda.smiles.transforms import AugmentTensor

from utils.hyperparams import ACTIVATION_FN_FACTORY, LOSS_FN_FACTORY
from utils.interpret import monte_carlo_dropout, test_time_augmentation
from utils.layers import (
    ContextAttentionLayer, convolutional_layer, dense_layer
)
from utils.utils import get_device, get_log_molar