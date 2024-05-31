
import sys
import os


from tntorch_exps import noise_exps_tntorch

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
                'target': 'foot',
                'max_ranks': [292,195]  # Updated from the 256**3 row under TT
            },
        }

    # take part before number - e.g. girl16k is tokoyo
    class_name ="isotropic"
    args = Config()
    args.dimensions = 3
    args.payload_position = "grayscale"
    args.payload = 0
    args.dtype = "float32"

    noise_exps_tntorch(configurations, args)