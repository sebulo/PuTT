
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
        "isotropic256": {
            'target': 'isotropic256',
            'max_ranks': [292, 195, 102]  # Updated from the 256**3 row under TT
        },
        "isotropic128": {
            'target': 'isotropic128',
            'max_ranks': [105, 64, 40]  # Updated from the 128**3 row under TT
        },
        "isotropic64": {
            'target': 'isotropic64',
            'max_ranks': [80, 40, 20]  # Updated from the 64**3 row under TT
        },
         "isotropic512": {
            'target': 'isotropic512',
            'max_ranks': [350, 250, 150]  # Updated from the 512**3 row under TT
        },
        "isotropic1024": {
            'target': 'isotropic1024',
            'max_ranks': [250, 420, 300]  # Updated from the 1024**3 row under TT
        },  
        }


    class_name ="flower"
    args = Config()
    args.dimensions = 3
    args.payload_position = "grayscale"
    args.payload = 0
    args.dtype = "float32"

    exps_tntorch(configurations, args)