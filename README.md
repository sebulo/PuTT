# Coarse-To-Fine Tensor Trains for Compact Visual Representations
## [Project page](https://sebulo.github.io/PuTT_website/) |  [Paper](https://link-to-paper)
This repository contains a pytorch implementation for the paper: [TensoRF: Tensorial Radiance Fields](https://arxiv.org/abs/2203.09517). Our work present a novel approach to model and reconstruct radiance fields, which achieves super
**fast** training process, **compact** memory footprint and **state-of-the-art** rendering quality.<br><br>


This repository offers a PyTorch implementation of the "Coarse-To-Fine Tensor Trains for Compact Visual Representations" method, described in our [paper](https://link-to-paper). It features our Prolongation Upsampling Tensor Train (PuTT) approach, designed for learning compact Quantized Tensor Train representations of large dimensional tensors, e.g. images, volumes.

## Repository Structure
The repository is organized into two main components:
1. **PuTT**: This is the core implementation of the PuTT method applied to 2D and 3D representations.
2. **PuTT NeRF**: This extends the PuTT method to Novel View Synthesis, integrating elements from [TensoRF](https://apchenstu.github.io/TensoRF/) and [TTNF](https://www.obukhov.ai/ttnf).

For an in-depth understanding of the PuTT method, refer to our [paper](https://link-to-paper).

### Installation

**Environment Requirements:**
Install environment:
```
conda env create -f environment.yml
```
This will create a conda environment with all the necessary packages. Make sure to activate the environment:
```
conda activate putt
```


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

## License

This software is released under the MIT License. You can view a license summary [here](LICENSE).

Portions of source code are taken from external sources under different licenses, including the following:
- [NeRF](https://github.com/yenchenlin/nerf-pytorch) (MIT)
- [TensoRF](https://apchenstu.github.io/TensoRF/) (MIT)
- [TTNF](https://www.obukhov.ai/ttnf) (CC BY-NC 4.0 DEED) - Please note that this portion of the software remains under CC BY-NC 4.0 and is not covered by the MIT License.


## Acknowledgements
This work was supported by Danish Data Science Academy, which is funded by the Novo Nordisk Foundation (NNF21SA0069429) and VILLUM FONDEN (40516).
