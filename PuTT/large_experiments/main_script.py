import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)

import itertools
import configargparse
import logging

from train import train
from utils import check_validity_upsampling_steps, get_kwargs_dict  # Import required function

import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if not os.environ.get('WANDB_API_KEY'):
        logger.error("WANDB_API_KEY not set in the environment")
        sys.exit(1)


def run_train(config_file):
    setup_environment()

    args = get_kwargs_dict(config_file)

    # end_reso is needed to fill in value for init_reso_arr if it is empty 
    end_reso = args.end_reso

    # if batch_size_arr or lr_arr is empty, use max_batch_size or lr
    batch_sizes = getattr(args, 'batch_size_arr', [])
    if not batch_sizes:
        batch_sizes = [args.max_batch_size]

    learning_rates = getattr(args, 'lr_arr', [])
    if not learning_rates:
        learning_rates = [args.lr]

    num_iterations_arr = getattr(args, 'num_iterations_arr', [])
    if len(num_iterations_arr) == 0:
        num_iterations_arr = [args.num_iterations]

    subset_to_train_on_arr = getattr(args, 'subset_to_train_on_arr', [])
    if len(subset_to_train_on_arr) == 0:
        subset_to_train_on_arr = [args.subset_to_train_on]
        
    regularization_weight_arr = getattr(args, 'regularization_weight_arr', [])
    if len(regularization_weight_arr) == 0:
        regularization_weight_arr = [args.regularization_weight]
        
    rank_upsampling_rank_range_arr = getattr(args, 'rank_upsampling_rank_range_arr', [])
    if len(rank_upsampling_rank_range_arr) == 0:
        rank_upsampling_rank_range_arr = [[]]
    rank_upsampling_iteration_range_arr = getattr(args, 'rank_upsampling_iteration_range_arr', [])
    if len(rank_upsampling_iteration_range_arr) == 0:
        rank_upsampling_iteration_range_arr =  [[]]
        
    sigma_init_arr = getattr(args, 'sigma_init_arr', [])
    if not sigma_init_arr:
        sigma_init_arr = [args.sigma_init]
        


    parameter_combinations = itertools.product(
        getattr(args, 'seeds', [0]),
        getattr(args, 'init_reso_arr', [end_reso]), 
        getattr(args, 'max_ranks_arr', [args.max_rank]),
        getattr(args, 'tilting_mode_arr', [args.tilting_mode]),
        getattr(args, 'tilt_angle_arr', [args.tilt_angle]),
        getattr(args, 'noise_type_arr', ['None']),
        getattr(args, 'noise_std_arr', [0.0]),
        getattr(args, 'iterations_for_upsampling_arr', args.iterations_for_upsampling),
        batch_sizes,
        learning_rates,
        num_iterations_arr,
        subset_to_train_on_arr,
        regularization_weight_arr,
        rank_upsampling_rank_range_arr,
        rank_upsampling_iteration_range_arr,
        sigma_init_arr
        
    )

    print("##################")
    print("##################")
    print("Noise_type_arr", args.noise_type_arr)
    print("Noise_std_arr", args.noise_std_arr)
    print("Max_ranks_arr", args.max_ranks_arr)
    print("Seeds", args.seeds)
    print("Init_reso_arr", args.init_reso_arr)
    print("Tilt_angle_arr", args.tilt_angle_arr)
    print("Iterations_for_upsampling", args.iterations_for_upsampling_arr)
    print("Batch_size_arr", args.batch_size_arr)
    print("Lr_arr", args.lr_arr)
    print("Num_iterations_arr", args.num_iterations_arr)
    print("Subset_to_train_on_arr", args.subset_to_train_on_arr)
    print("Regularization_weight_arr", args.regularization_weight_arr)
    print("Rank_upsampling_rank_range_arr", args.rank_upsampling_rank_range_arr)
    print("Rank_upsampling_iteration_range_arr", args.rank_upsampling_iteration_range_arr)
    print("sigma_init_arr", args.sigma_init_arr)
    
    print("##################")
    print("##################")

    for seed, init_reso, max_rank, tilting_mode, tilt_angle, noise_type, noise_std, iterations_for_upsampling, batch_size, lr, num_iterations, subset_to_train_on, regularization_weight, rank_upsampling_rank_range, rank_upsampling_iteration_range,sigma_init in parameter_combinations:
        print(f"Starting for Seed: {seed},init_reso: {init_reso}, max_rank: {max_rank}, tilting_mode: {tilting_mode}, tilt_angle: {tilt_angle}, \n noise_type: {noise_type}, noise_std: {noise_std}, iterations_for_upsampling: {iterations_for_upsampling}, batch_size: {batch_size}, lr: {lr}")
        print(f"Regularization weight: {regularization_weight}")
        print(f"Rank upsampling rank range: {rank_upsampling_rank_range}")
        print(f"Rank upsampling iteration range: {rank_upsampling_iteration_range}")
        args.seed = seed
        args.init_reso = init_reso
        args.max_rank = max_rank
        args.tilting_mode = tilting_mode
        args.tilt_angle = tilt_angle
        args.noise_type = noise_type
        args.noise_std = noise_std
        args.max_batch_size = batch_size
        args.lr = lr
        args.iterations_for_upsampling = iterations_for_upsampling
        args.num_iterations = num_iterations
        args.subset_to_train_on = subset_to_train_on
        args.sigma_init = sigma_init
        
        args.rank_upsampling_rank_range = rank_upsampling_rank_range
        args.rank_upsampling_iteration_range = rank_upsampling_iteration_range

        args.regularization_weight = regularization_weight
        
        # Check if upsampling iterations are valid
        is_valid = check_validity_upsampling_steps(init_reso, end_reso, iterations_for_upsampling, args.num_iterations)
        if not is_valid:
            logger.error(" ##### !!!! ##### Invalid upsampling iterations for {} to {} with {} iterations and {} num iterations".format(init_reso, end_reso, iterations_for_upsampling, args.num_iterations))
            # skip this combination
            continue
        elif noise_type == 'None' and noise_std != 0.0 or noise_type != 'None' and noise_std == 0.0:
            # skip this combination
            logger.error(" ########## Skipping combination with noise_type: {} and noise_std: {}".format(noise_type, noise_std))
            continue


            
        train(args)



if __name__ == "__main__":
    parser = configargparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args, unknown = parser.parse_known_args()

    run_train(args.config)
