import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument("--add_timestamp", type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')

    parser.add_argument('--with_depth', action='store_true')
    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)

    parser.add_argument('--model_name', type=str, default='TensorVMSplit',
                        choices=['TensorVMSplit', 'TensorCP', 'TensorTT', 'TensorRing'])

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000)

    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'nsvf', 'dtu','tankstemple', 'own_data'])


    # training options
    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.02,
                        help='learning rate')    
    parser.add_argument("--lr_init_shader", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help = 'number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')
    parser.add_argument("--lr_min_factor", type=float, default=0,
                        help='minimum learning rate')

    # loss
    parser.add_argument("--L1_weight_inital", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0,
                        help='loss weight')
    parser.add_argument("--Ortho_weight", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_density", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=0.0,
                        help='loss weight')
    
    
    # fused
    parser.add_argument("--fused", type=int, default=0, help='use fused model QTT')

    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument("--init_scale", type=float, default=0.1)

    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25,
                        help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')
                        
    # network decoder
    parser.add_argument("--shadingMode", type=str, default="MLP_PE",
                        help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=6,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=6,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')
    
    # rank upsampling #
    parser.add_argument("--upsample_ranks_iterations", type=int, action="append")
    parser.add_argument("--upsample_ranks_app", type=int, action="append")
    parser.add_argument("--upsample_ranks_density", type=int, action="append")
    parser.add_argument("--rank_upsampling_lr_factor", type=float, default=1.0, help='lr factor to adjust current lr after rank upsampling')

    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)
    
    # Tensor Ring use_channel_comp_density
    parser.add_argument("--use_channel_comp_density", type=int, default=0)
    # is_tensor_ring
    parser.add_argument("--is_tensor_ring", type=int, default=1)
    

    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default='softplus')
    parser.add_argument('--ndc_ray', type=int, default=0)
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)
    #args.should_shrink
    parser.add_argument('--should_shrink', type=int, default=0)


    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    parser.add_argument('--N_voxel_init',
                        type=int,
                        default=100**3)
    parser.add_argument('--N_voxel_final',
                        type=int,
                        default=300**3)
    parser.add_argument("--upsample_list", type=int, action="append")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")

    parser.add_argument('--idx_view',
                        type=int,
                        default=0)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5,
                        help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=10000,
                        help='frequency of visualize the image')
    
    # seed
    parser.add_argument("--seed", type=int, default=20211202,
                        help='random seed')


    #i_wandb = 1
    parser.add_argument('--i_wandb', type=int, default=0,
                        help ="use wandb for exp tracking")
    # use_TTNF_sampling
    parser.add_argument('--use_TTNF_sampling', type=int, default=0,
                        help ="use V2 sampling for TTNF")

    # input_image_normalization - type None, mean_std, min_max
    parser.add_argument('--input_image_normalization', type=str, default='None',
                        help ="input image normalization")

    #plot_after_upsampling
    parser.add_argument('--plot_after_upsampling', type=int, default=0,
                        help ="to test just after upsampling for testing purposes")
    #max_rank
    parser.add_argument('--max_rank', type=int, default=256,
                        help ="max rank for TTNF")
    #max_rank
    parser.add_argument('--max_rank_density', type=int, default=-1,
                        help ="max rank for sigma TT")
    
    parser.add_argument('--max_rank_appearance', type=int, default=-1,
                        help ="max rank for appearance TT")
    
    parser.add_argument('--warmup_steps', type=int, default=0, help='number of lr warmup steps')

    parser.add_argument('--canonization', type=str, default = 'None', choices=['None', 'left', 'right', 'middle'], help='canonize the TT cores')

    parser.add_argument('--compression_alg', type=str, default='compress_all', choices=['compress_all', 'tree_compress', 'mps_compress'], 
                        help='compression algorithm for truncating the TT ranks')


    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()