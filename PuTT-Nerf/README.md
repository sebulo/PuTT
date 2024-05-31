# PuTT NeRF
## [Project page](https://sebulo.github.io/PuTT_website/) |  [Paper](https://link-to-paper)

This repository contains a pytorch implementation for our Prolongation Upsampling Tensor Train (PuTT) method applied to Novel View Synthesis .<br><br>
## Installation
Use the install guide from the preivous section to make a Conda PuTT environment. 

## Dataset
* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [Synthetic-NSVF](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip)
* [Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)
* [Forward-facing](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)



## Quick Start
The training script is in `train.py`, to train a PuTT:

```
python train.py --config configs/PuTT_hotdog.txt
```


we provide a few examples in the configuration folder, please note:

 `dataset_name`, choices = ['blender', 'llff', 'nsvf', 'tankstemple'];

 `shadingMode`, choices = ['MLP_Fea', 'SH'];

 `model_name`, choices = ['TensorTT', TensorVMSplit', 'TensorCP'], corresponding to the Tensor Train, the VM and CP decomposition. 
 You need to uncomment the last a few rows of the configuration file if you want to training with the TensorCP model；

 'max_rank' is paramter for the rank of the Tensor Train.

`fused` is a flag to use the fused version of the model, where both appearance and density is modeled by the same tensor. With fused set to 0, the model will use separate tensors for appearance and density. This enables the use of different ranks for the two tensors, which can be useful for some datasets, e.g. `max_rank_appearance: 250` and `max_rank_density: 150`.

 'use_TTNF_sampling' is paramter for choosing V2 sampling as used by TT-NF.

 'upsample_list' determines iterations to perform upsampling.

 `n_lamb_sigma` and `n_lamb_sh` are string type refer to the basis number of density and appearance along XYZ
dimension;

 `N_voxel_init` and `N_voxel_final` control the resolution of the grid, which is initialized with `N_voxel_init` and
 upsampled to `N_voxel_final` in number of steps equal to the length of `upsample_list`. For QTT the grid dim should be powers of 2., e.g. `N_voxel_init = 32**3` and `N_voxel_final = 256**3`;

 `N_vis` and `vis_every` control the visualization during training;


  
You need to set `--render_test 1`/`--render_path 1` if you want to render testing views or path after training. 

More options refer to the `opt.py`. 


## Rendering

```
python train.py --config configs/lego.txt --ckpt path/to/your/checkpoint --render_only 1 --render_test 1 
```

You can just simply pass `--render_only 1` and `--ckpt path/to/your/checkpoint` to render images from a pre-trained
checkpoint. You may also need to specify what you want to render, like `--render_test 1`, `--render_train 1` or `--render_path 1`.
The rendering results are located in your checkpoint folder. 

## Extracting mesh
You can also export the mesh by passing `--export_mesh 1`:
```
python train.py --config configs/lego.txt --ckpt path/to/your/checkpoint --export_mesh 1
```
Note: Please re-train the model and don't use the pretrained checkpoints provided by us for mesh extraction, 
because some render parameters has changed.


## Logging
The code performs logging to the console, tensorboard file in the experiment log directory, and also [Weights and Biases](https://www.wandb.com). Upon the first run, please enter your account credentials, which can be obtained by registering a free account with the service.


## Training with your own data
We provide two options for training on your own image set:

1. Following the instructions in the [NSVF repo](https://github.com/facebookresearch/NSVF#prepare-your-own-dataset), then set the dataset_name to 'tankstemple'.
2. Calibrating images with the script from [NGP](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md):
`python dataLoader/colmap2nerf.py --colmap_matcher exhaustive --run_colmap`, then adjust the datadir in `configs/your_own_data.txt`. Please check the `scene_bbox` and `near_far` if you get abnormal results.
    

## Citation
If you find our code or paper helps, please consider citing:
```
@INPROCEEDINGS{loeschckePuTT,
  author = {Sebastian Loeschcke and Dan Wang and Christian Leth-Espensen and Serge Belongie and Michael Kastoryano and Sagie Benaim},
  title = {Coarse-To-Fine Tensor Trains for Compact Visual Representations},
  booktitle = {},
  year = {2023}
}
```

## Acknowledgements
This work was supported by Danish Data Science Academy, which is funded by the Novo Nordisk Foundation (NNF21SA0069429) and VILLUM FONDEN (40516).
