
import sys
import os

from tntorch_exps import exps_tntorch
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src')))

from train import train
from opt import *
from utils import *

from tn_utils import *

import argparse

import numpy as np

import os
import itertools
from subprocess import run
import wandb

import pandas as pd
import time

import numpy as np
import tntorch as tn
import torch
from PIL import Image
import matplotlib.pyplot as plt



if __name__ == '__main__':

    configurations = {
            "tokyo8k": {
                'target': 'tokyo8k',
                'max_ranks':  [350, 250, 150]
            },
            "tokyo4k": {
                'target': 'tokyo4k',
                'max_ranks': [300,200,100]
            },
            "tokyo2k": {
                'target': 'tokyo2k',
                'max_ranks':  [250,150,75]
            },
            "tokyo1k": {
                'target': 'tokyo1k',
                'max_ranks':[200,100,50]
            },
            "tokyo16k": {
                'target': 'tokyo16k',
                'max_ranks': [400, 300, 200]
            },
        }

    # take part before number - e.g. tokyo16k is tokoyo
    class_name ="tokyo"
    args = Config()
    args.dimensions = 2
    args.payload_position = "grayscale"
    args.payload = 0
    args.dtype = "float32"

    exps_tntorch(configurations, args, class_name)

