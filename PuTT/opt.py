
import yaml

class Config:
    def __init__(self):
        self.model = 'QTT'  # "CP,QTT,VM"
        self.base_config = 'configs/2d_base_config.yaml'  # "base config file path"
        self.exp_name = ''
        
        self.dimensions = 2
        self.target = 'cameraman'
        self.seeds = []  # "list of seeds to try" - if empty just use seed
        self.seed = 0  
        self.dtype = 'float32'
        self.use_TTNF_sampling = 0
        self.device_type = 'gpu'

        # Resolutions
        self.init_reso_arr = [32]  # "list of initial resolutions to try"
        self.init_reso = 32  # "list of initial resolutions to try"
        self.end_reso = 512
        self.iterations_for_upsampling_arr  = [[32, 64, 128, 256]]  # "List of lists of iterations for upsampling"
        self.iterations_for_upsampling = [32, 64, 128, 256]  # "List of lists of iterations for upsampling"
        self.max_ranks_arr = []  # list of max ranks to try - if empty just use max_rank
        self.max_rank = 256  # list of max ranks to try
        self.max_batch_size = 16384  # 128**2
        self.batch_size_arr = []  # list of batch sizes to try - if empty just use batch_size

        self.sigma_init = 0 # 0 means do calculation instead of giving sigma explicitly
        self.sigma_init_arr = []  # list of sigma_init to try - if empty just use sigma_init
        
        # Training
        self.use_tqdm = 0
        self.use_wandb = 0
        self.wandb_limited_logging = 0
        self.only_local_wandb = 0
        self.log_every = 50
        self.save_training_images = 1 # save training images  if save_images_locally_wandb then saved locally else saved in wandb
        self.save_images_locally_wandb = 1 # should images be saved locally for enabling later upload to wandb
        self.plot_3d_local = False

        # LR
        self.lr = 0.01
        self.lr_arr = []  # list of learning rates to try - if empty just use lr
        self.lr_decay_factor = 0.8
        self.lr_decay_factor_scheduler = 0.995
        self.lr_decay_factor_until_next_upsampling = 0.2

        # Save config
        self.save_end_results_locally = 0
        self.show_end_results_locally = 0
        self.get_noisy_target_psnr_measures = 0
        self.plot_upsampling = 0  # plot saved images after upsampling
        
        self.save_every = 2000
        self.save_every_start_iteration = 0
        self.plot_training = 0  # "plot while training or not at every "save_every" iteration" starting from "save_every_start_iteration"
        self.calculate_psnr_while_training = 1  # "calculate psnr while training or not"
        self.save_learned_recon = 0
        self.save_dir = 'saved_images'

        self.payload = 0  # "payload is 3 for RGB images,1 for grayscale images and X for X num features"
        self.payload_position = 'grayscale'  # "first_core, last_core, or grayscale"
        self.compute_reconstruction = 1  # "compute reconstruction or not - 0 if object is too big to fit in memory"
        self.loss_fn_str = 'L2'
        self.sample_with_replacement = 1
        self.sample_weights_converge_steps = 40
        self.activation = 'None'
        self.compression_alg = 'compress_all'  # "compress_all, tree_compress, None"
        self.regularization_type = 'None'  # "L1, L2, TV or None"
        self.regularization_weight = 0
        self.regularization_weight_arr = []
        self.num_iterations = 100000
        self.num_iterations_arr = []
        self.warmup_steps = 50
        self.canonization = 'left'  # "left, middle, right or none"
        
        # Rank upsampling
        self.rank_upsampling_rank_range = [] # [50, 200, 5] # [start, end, step] # range of ranks
        self.rank_upsampling_iteration_range = [] # [50, 200, 5] # [start, end, step] # range of iteration itervalues for upsampling
        self.rank_upsampling_rank_range_arr = [] # for doing batch experiments
        self.rank_upsampling_iteration_range_arr = [] # for doing batch experiments
            
        # TT is_tensor_ring
        self.is_tensor_ring = 1
        self.channel_rank = 30

        # Noise
        self.noise_type_arr = []  # "None, gaussian, laplace" - list of noise types to try - if empty just use noise_type
        self.noise_type = 'None'
        self.noise_std_arr = []  # list of noise stds to try - if empty just use noise_std
        self.noise_std = 0.0  
        self.noise_mean = 0.0 # only laplace

        # train on subset of data
        self.subset_to_train_on = 1.0 # between 0 and 1 indicating the fraction of the data to use - 1 - subset_to_train_on is the fraction of the data NOT used for training
        self.is_random_box_impainting  = 0 # random box subsample or not
        self.subset_to_train_on_arr = [] # list of fractions to try - if empty just use subset_to_train_on
        self.default_val_for_non_sampled = 0 # default value for pixels that are not sampled in the subset_to_train_on
        self.masked_avg_pooling = 0 # use masked avg pooling or not - meaning downsampled pixels that are 0 (meaning not sampled) are not counted in the average ---> no shift to 0 in the output
        self.plot_subsampled_target = 0 # plot the subsampled target in beginning
        
        self.factor_reduce_lr_based_on_noise = 0 # reduce lr based on noise_std/subset_to_train_on - if not 0 then lr is multiplied by this factor: lr = lr * factor_reduce_lr_based_on_noise ** (noise_std/subset_to_train_on)) 

        # Tilting
        self.tilt_angle_arr = []  # list of tilt angles to try - if empty just use tilt_angle
        self.tilt_angle = 0  # list of tilt angles to try
        self.tilting_mode = 0
        self.zero_padding_tilting = 0 # "zero padding when rotating/tilting or not - to avoid some tagets having more black pixels than others"

        # for Grid + MLP
        self.shading_mode = 'MLP'
        self.output_dim = 3
        self.fea_pe = 0
        self.featureC = 128
        self.lr_init_mlp = 0.001


        self.fill_empty_lists()

    def fill_empty_lists(self):
        if len(self.seeds) == 0:
            self.seeds = [self.seed]
        if len(self.max_ranks_arr) == 0:
            self.max_ranks_arr = [self.max_rank]
        if len(self.noise_type_arr) == 0:
            self.noise_type_arr = [self.noise_type]
        if len(self.noise_std_arr) == 0:
            self.noise_std_arr = [self.noise_std]
        if len(self.tilt_angle_arr) == 0:
            self.tilt_angle_arr = [self.tilt_angle]


def make_dict_from_args(args):
    return vars(args)

def yaml_config_parser(config_file):
    """
    Parse a YAML configuration file and return a dictionary with the configurations.
    :param config_file: Path to the configuration file.
    :return: A dictionary containing the configurations read from the file.
    """
    with open(config_file, 'r') as f:
        configs = yaml.safe_load(f)
    return configs
