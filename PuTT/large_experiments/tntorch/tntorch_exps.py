import sys
import os
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


def exps_tntorch(configurations, args, class_name = None):
    psnrs = []
    ssims = []
    errors = []
    times = []
    targets = []
    ranks = []

    for config_name, config in configurations.items():
        print(f"Starting for Configuration: {config_name}")
        args.target = config['target']
        target = load_target(args)
        target_org = target.clone()
        if class_name is None:
            class_name = config_name


        for rank in config['max_ranks']:
            print(f"Starting for Target {args.target} with rank {rank}")
            time_start = time.time()
            #tensorly_qtt
            target_recov, target_org, tn_org = qtt_svd(target, rank, dim=args.dimensions,
                                                            payload_dim=args.payload,
                                                            payload_position=args.payload_position,
                                                            library="tntorch")
                                                            # library="tensorly")

            error = torch.mean((target_org - target_recov)**2)
            psnr_val = 10 * torch.log10(1 / error)
            #psnr_val = psnr_with_mask(target_org, target_recov)
            ssim_val = calculate_ssim(target_org, target_recov, args.payload, args.dimensions, patched=False)
            

            time_end = time.time()
            time_elapsed = time_end - time_start
            print(f"Time elapsed: {time_elapsed}")
            print(f"PSNR: {psnr_val}")
            print(f"SSIM: {ssim_val}")
            print(f"Error: {error}")
            times.append(time_elapsed)
            # check if item() is needed
            if torch.is_tensor(psnr_val):
                psnrs.append(psnr_val.item())
            else:
                psnrs.append(psnr_val)
            if torch.is_tensor(ssim_val):
                ssims.append(ssim_val.item())
            else:
                ssims.append(ssim_val)

            errors.append(error)
            targets.append(config_name)
            ranks.append(rank)
    

    df = pd.DataFrame(list(zip(psnrs, ssims, errors, times, targets, ranks)), columns =['PSNR', 'SSIM', 'error', 'time', 'target', 'rank'])
    df.to_csv("tntorch_qtt_{}.csv".format(class_name), index=False)
    print(df)

def noise_exps_tntorch(configurations, args, class_name = None):
    psnrs = []
    ssims = []
    errors = []
    times = []
    targets = []
    noise_args = []
    noise_types_arr = []
    ranks = []

    for config_name, config in configurations.items():
        if class_name is None:
            class_name = config_name
        print(f"Starting for Configuration: {config_name}")
        args.target = config['target']
        target = load_target(args)
        target_org_org = target

        noise_stds = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3,0.4,0.5,1.0,3.0,10.0]
        noise_types = ["None", "gaussian", "laplace"]
        
        for noise_std in noise_stds:
            for noise_type in noise_types:
                args.noise_std = noise_std
                args.noise_type = noise_type
                
                if (noise_type == "None" and noise_std > 0.0) or (noise_type != "None" and noise_std == 0.0):
                    continue
                    
                if args.noise_std > 0.0 and args.noise_type != "None":
                    print("using noisy target")
                    noisy_target = make_noisy_target(target, args)
                else:
                    noisy_target = target

                for rank in config['max_ranks']:
                    print(f"Starting for Target {args.target} with rank {rank}, noise_std {args.noise_std}, noise_type {args.noise_type}")

                    time_start = time.time()
                    # target_recov, target_org, tn_org = tntorch_qtt(noisy_target, rank, dim=args.dimensions,
                    #                                                 payload_dim=args.payload,
                    #                                                 payload_position=args.payload_position)

                    #tensorly_qtt
                    target_recov, target_org, tn_org = qtt_svd(noisy_target, rank, dim=args.dimensions,
                                                                    payload_dim=args.payload,
                                                                    payload_position=args.payload_position,
                                                                    library="tntorch")
                                                                    # library="tensorly")

                    error = torch.mean((target_org_org - target_recov) ** 2)
                    psnr_val = psnr(target_org_org, target_recov)
                    ssim_val = calculate_ssim(target_org_org, target_recov, args.payload, args.dimensions, patched=False)

                    time_end = time.time()
                    time_elapsed = time_end - time_start
                    print(f"Time elapsed: {time_elapsed}")
                    print(f"PSNR: {psnr_val}")
                    print(f"SSIM: {ssim_val}")
                    times.append(time_elapsed)
                    if torch.is_tensor(psnr_val):
                        psnrs.append(psnr_val.item())
                    else:
                        psnrs.append(psnr_val)
                    if torch.is_tensor(ssim_val):
                        ssims.append(ssim_val.item())
                    else:
                        ssims.append(ssim_val)

                    errors.append(error)
                    targets.append(config_name)
                    noise_args.append(noise_std)
                    noise_types_arr.append(noise_type)
                    ranks.append(rank)
                    print("tn_org", tn_org)

    df = pd.DataFrame(list(zip(psnrs, ssims, errors, times, targets, noise_args, noise_types_arr, ranks)),
                      columns=['PSNR', 'SSIM', 'MSE', 'time', 'target', 'noise_std', 'noise_type', 'rank'])
    df.to_csv("tntorch_qtt_noise_{}.csv".format(class_name), index=False)
    print(df)


def tilting_exps_tntorch(configurations, args, class_name = None):
    psnrs = []
    ssims = []
    errors = []
    times = []
    targets = []
    ranks = []
    tilt_angles_arr = []
    zero_padding_arr = []

    tilt_angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]
    tilting_mode = 1
    zero_padding = args.zero_padding_tilting
    print("zero_padding", zero_padding)

    for tilt_angle in tilt_angles:

        for config_name, config in configurations.items():
            if class_name is None:
                class_name = config_name
            print(f"Starting for Configuration: {config_name}")
            args.target = config['target']
            target = load_target(args)
            target_org = target.clone()
            zero_padding = args.zero_padding_tilting
            target = make_rotation_of_target(target, tilt_angle, num_dims = args.dimensions, zero_padding = zero_padding)
            

            for rank in config['max_ranks']:
                print(f"Starting for Target {args.target} with rank {rank}")
                time_start = time.time()
                #tensorly_qtt
                target_recov, target_org, tn_org = qtt_svd(target, rank, dim=args.dimensions,
                                                                payload_dim=args.payload,
                                                                payload_position=args.payload_position,
                                                                library="tntorch")
                                                                # library="tensorly")

                error = torch.mean((target_org - target_recov)**2)
                psnr = 10 * torch.log10(1 / error)
                psnr_val = psnr_with_mask(target_org, target_recov)
                ssim_val = calculate_ssim(target_org, target_recov, args.payload, args.dimensions, patched=False)
                

                time_end = time.time()
                time_elapsed = time_end - time_start
                print(f"Time elapsed: {time_elapsed}")
                print(f"PSNR: {psnr}")
                print(f"PSNR val: {psnr_val}")
                print(f"SSIM: {ssim_val}")
                print(f"Error: {error}")
                times.append(time_elapsed)
                if torch.is_tensor(psnr_val):
                    psnrs.append(psnr_val.item())
                else:
                    psnrs.append(psnr_val)
                if torch.is_tensor(ssim_val):
                    ssims.append(ssim_val.item())
                else:
                    ssims.append(ssim_val)
                errors.append(error)
                tilt_angles_arr.append(tilt_angle)
                targets.append(config_name.split("_")[0])
                ranks.append(rank)
                zero_padding_arr.append(zero_padding)
            
        #make csv

    df = pd.DataFrame(list(zip(psnrs, ssims, errors, times, targets, ranks, tilt_angles_arr, zero_padding_arr)), columns =['PSNR', 'SSIM', 'Error', 'time', 'target', 'rank', 'tilt_angle', 'zero_padding'])
    df.to_csv("tntorch_qtt_{}_tilting_0pad_{}.csv".format(class_name, zero_padding), index=False)
    print(df)