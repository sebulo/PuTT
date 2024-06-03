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


We provide a few examples in the configuration folder. Please note the following parameters:

- `dataset_name`: choices = ['blender', 'llff', 'nsvf', 'tankstemple'];
- `shadingMode`: choices = ['MLP_Fea', 'SH'];
- `model_name`: choices = ['TensorTT', 'TensorVMSplit', 'TensorCP'], corresponding to the Quantized Tensor Train, the VM, and CP decomposition.
- `max_rank`: parameter for the rank of the Tensor Train.
- `fused`: flag to use the fused version of the model, where both appearance and density are modeled by the same tensor. With `fused` set to 0, the model will use separate tensors for appearance and density, allowing different ranks for the two tensors, which can be useful for some datasets, e.g., `max_rank_appearance: 250` and `max_rank_density: 150`.
- `use_TTNF_sampling`: parameter for choosing V2 sampling as used by TT-NF.
- `upsample_list`: determines iterations to perform upsampling.
- `update_AlphaMask_list`: specifies iterations to update the alpha mask.
- `n_lamb_sigma` and `n_lamb_sh`: string type parameters referring to the basis number of density and appearance along the XYZ dimension.
- `N_voxel_init`: controls the initial resolution of the grid (e.g., `N_voxel_init = 4096` for a 16³ grid).
- `N_voxel_final`: controls the final resolution of the grid (e.g., `N_voxel_final = 16777216` for a 256³ grid). The grid is upsampled in steps equal to the length of `upsample_list`. For QTT, the grid dimensions should be powers of 2.
- `N_vis`: controls the number of visualizations during training.
- `vis_every`: specifies the frequency (in iterations) for visualizations.
- `render_test`: set to 1 to render testing views after training.
- `max_rank_appearance`: maximum rank for the appearance tensor (default: 280).
- `max_rank_density`: maximum rank for the density tensor (default: 200).

You need to set `--render_test 1`/`--render_path 1` if you want to render testing views or paths after training. 

For more options, refer to `opt.py`. 


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
