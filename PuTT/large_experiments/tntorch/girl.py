
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
            "girl8k": {
                'target': 'girl8k',
                'max_ranks':  [350, 250, 150]
            },
            "girl4k": {
                'target': 'girl4k',
                'max_ranks': [300,200,100]
            },
            "girl2k": {
                'target': 'girl2k',
                'max_ranks':  [250,150,75]
            },
            "girl1k": {
                'target': 'girl1k',
                'max_ranks':[200,100,50]
            },
            "girl16k": {
                'target': 'girl16k',
                'max_ranks': [400, 300, 200]
            },
        }

       # take part before number - e.g. girl16k is tokoyo
    class_name ="girl"
    args = Config()
    args.dimensions = 2
    args.payload_position = "grayscale"
    args.payload = 0
    args.dtype = "float32"

    exps_tntorch(configurations, args)