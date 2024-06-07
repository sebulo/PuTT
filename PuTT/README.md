# PuTT
## [Project page](https://sebulo.github.io/PuTT_website/) |  [Paper](https://link-to-paper)
This repository contains a pytorch implementation for Coarse-To-Fine Tensor Trains for Compact Visual Representations and our method Prolongation Upsampling Tensor Train (PuTT).<br>

### Installation
Use the install guide from the preivous section to make a Conda PuTT environment. 

## Datasets
* [Images-16k](https://drive.google.com/drive/folders/157jxhKVT1ssu3VivrO0jC4SAVGjFRv_J?usp=sharing) 
* [John Hopkins Turbulence 3D Data ](https://turbulence.pha.jhu.edu/)
* [ETH Reasearch 3D data](https://www.ifi.uzh.ch/en/vmml/research/datasets.html)

## Downsampling images and 3D data
### Downsampling Images
To downsample image files (.jpg, .png, .npy), specify the directory containing the images and the target resolutions as comma-separated values. For example:
```
python get_data.py --data_dir path/to/image_data --resolutions 1024,2048,4096,8192
```
### 3D Data
For downsampling 3D data in ".raw" format, in addition to the directory and resolutions, you must also specify the dimensions and data type of the raw files. For example:
```
python get_data.py --data_dir path/to/3d_data --resolutions 64,128,256,512 --raw_dims X,Y,Z --raw_dtype uint8
```
### Parameters
- **--data_dir**: Directory containing the data files.
- **--resolutions**: Comma-separated list of target resolutions (e.g., 32,64).
- **--raw_dims**: (Only for 3D raw data) Comma-separated dimensions of the raw files (e.g., 256,256,256).
- **--raw_dtype**: (Only for 3D raw data) Data type of the raw files (uint8 or uint16).



## Coarse to Fine Training of Quantized Tensor Train Models

Run the PuTT model on the Image Girl1k example by using the following command:
```
python train.py --config configs/girl1k_QTT.yaml
```
### Configuration File Details

Customize your training by modifying the `girl1k_QTT.yaml` configuration file. Key parameters include:

- **target**: The path to your target image.
  - `target: data/images/girl_downsampled_images/girl_1k.png`

- **base_config**: The default configuration template for training. Variants are available for different data types (2D, 3D) and conditions (noisy, incomplete data).
  - `base_config: configs/base_configs/2d_base_config.yaml`

- **model**: Choose the model type. Options are QTT, Tucker, CP, and VM (VM is only applicable for 3D).
  - `model: QTT`

- **init_reso and end_reso**: Define the starting and final resolutions for the coarse to fine training process.
  - `init_reso: 64` 
  - `end_reso: 1024` 

- **iterations_for_upsampling**: A list of iterations where upsampling occurs. This should align with the resolution doubling steps from the initial to the end resolution. E.g for starting at 64 and ending at 1024, the length of the `iteartions_for_upsampling` should be 4:
  - `iterations_for_upsampling: [50, 100, 200, 400]`


- **max_rank**: The maximum tensor rank allowed for the model.
  - `max_rank: 200`

- **show_end_results_locally**: Set this to 1 to display intermediate results after training.
  - `show_end_results_locally: 1`

- **num_iterations**: The total number of training iterations.
  - `num_iterations: 600`

For more detailed information on each configuration option, refer to `src/opt.py`.


## Logging
The code performs logging to the console (if use_tqdm is True) and also [Weights and Biases](https://www.wandb.com) (if use_wandb is True). Upon the first run, please enter your account credentials, which can be obtained by registering a free account with the service.



## Citation
If you find our code or paper helps, please consider citing:
```
@misc{loeschcke2024coarsetofine,
      title={Coarse-To-Fine Tensor Trains for Compact Visual Representations}, 
      author={Sebastian Loeschcke and Dan Wang and Christian Leth-Espensen and Serge Belongie and Michael J. Kastoryano and Sagie Benaim},
      year={2024},
      eprint={2406.04332},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgements
This work was supported by Danish Data Science Academy, which is funded by the Novo Nordisk Foundation (NNF21SA0069429) and VILLUM FONDEN (40516)
