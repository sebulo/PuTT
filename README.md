# Coarse-To-Fine Tensor Trains for Compact Visual Representations
## [Project page](https://sebulo.github.io/PuTT_website/) |  [Paper]([https://arxiv.org/abs/2406.04332])

This repository offers a PyTorch implementation of the "Coarse-To-Fine Tensor Trains for Compact Visual Representations" method, described in our [paper](https://arxiv.org/abs/2406.04332) and published in the proceedings of the International Conference on Machine Learning (ICML) 2024. It features our Prolongation Upsampling Tensor Train (PuTT) approach, designed for learning compact Quantized Tensor Train representations of large dimensional tensors, e.g., images, and volumes.

## Repository Structure
The repository is organized into two main components:
1. **PuTT**: This is the core implementation of the PuTT method applied to 2D and 3D representations.
2. **PuTT NeRF**: This extends the PuTT method to Novel View Synthesis, integrating elements from [TensoRF](https://apchenstu.github.io/TensoRF/) and [TTNF](https://www.obukhov.ai/ttnf).

For an in-depth understanding of the PuTT method, refer to our [paper](https://arxiv.org/abs/2406.04332).

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
@misc{loeschcke2024coarsetofine,
      title={Coarse-To-Fine Tensor Trains for Compact Visual Representations}, 
      author={Sebastian Loeschcke and Dan Wang and Christian Leth-Espensen and Serge Belongie and Michael J. Kastoryano and Sagie Benaim},
      year={2024},
      eprint={2406.04332},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
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
