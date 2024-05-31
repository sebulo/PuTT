# Based on TensoRF code from https://github.com/apchenstu/TensoRF
# Modifications and/or extensions have been made for specific purposes in this project.

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tn_utils import *
from utils import SimpleSamplerNonRandom
from functools import reduce
import time 

from tqdm import tqdm

from model.BaseTNModel import BaseTNModel

class CPModel(BaseTNModel):
    def __init__(self, target, init_reso, max_rank=50, dtype='float32', loss_fn_str="L2", use_TTNF_sampling=False, payload=0, payload_position='first_core', canonization="first", activation="None", compression_alg="compress_all", regularization_type="TV", dimensions=2, regularization_weight=0.0, noisy_target = None, device = 'cpu', masked_avg_pooling=False):
        super().__init__(target, init_reso, max_rank, dtype, loss_fn_str, use_TTNF_sampling, payload, payload_position, canonization, activation, compression_alg, regularization_type, dimensions, noisy_target, device, masked_avg_pooling)
        # Additional initialization specific to TNmodeCP
        self.model = "CP"

        self.scale_init = 0.2
        self.vecMode = list(range(dimensions))[::-1]
        self.density_n_comp = [max_rank for i in range(dimensions)]
        
        self.regularization_weight = regularization_weight

        self.init_tn()
        self.set_compression_variables()



    def init_tn(self):
        self.density_line = self.init_one_svd(self.density_n_comp, self.grid_size, self.scale_init, self.device)

        if self.payload > 1:
            self.basis_mat = torch.nn.Linear(self.density_n_comp[0], self.payload, bias=False).to(self.device)


    def init_one_svd(self, n_component, grid_size, scale, device):
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], grid_size[vec_id], 1))))
        return torch.nn.ParameterList(line_coef).to(device)


    def compute_densityfeature(self, samples):
        if self.dimensions ==2:
            return self.compute_densityfeature2d(samples)
        elif self.dimensions ==3:
            return self.compute_densityfeature3d(samples)
        else:
            raise NotImplementedError



    def compute_densityfeature2d(self, xy_sampled):
        # xy_sampled has shape: (batch_size, 2)
        coordinate_line = torch.stack((xy_sampled[..., self.vecMode[0]], xy_sampled[..., self.vecMode[1]])) # coordinate_line has shape: (2, batch_size)
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(2, -1, 1, 2) # coordinate_line has shape: (2, batch_size, 1, 2)        
        line_coef_point = F.grid_sample(self.density_line[0], coordinate_line[[0]], align_corners=True).view(-1, *xy_sampled.shape[:1]) # line_coef_point has shape: (batch_size, 1)
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[1], coordinate_line[[1]], align_corners=True).view(-1, *xy_sampled.shape[:1]) # line_coef_point has shape: (batch_size, 1)
        if self.payload > 1:
            sigma_feature = self.basis_mat(line_coef_point.T)
        else:
            sigma_feature = torch.sum(line_coef_point, dim=0) # sigma_feature has shape: (1)
        
        return sigma_feature


    def compute_densityfeature3d(self, xyz_sampled):
        # xyz_sampled has shape: (batch_size, 3)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]])) # coordinate_line has shape: (3, batch_size)
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        line_coef_point = F.grid_sample(self.density_line[0], coordinate_line[[0]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[1], coordinate_line[[1]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[2], coordinate_line[[2]], align_corners=True).view(-1, *xyz_sampled.shape[:1])        
        sigma_feature = torch.sum(line_coef_point.view(-1, xyz_sampled.shape[0]), dim=0)  # sigma_feature has shape: (batch_size)      

        return sigma_feature


    def get_optparam_groups(self, lr_init = 0.02):
        out = []
        if self.payload > 1:
            out.append({'params': self.basis_mat.weight, 'lr': lr_init})
        for i in range(len(self.vecMode)):
            out.append({'params': self.density_line[i], 'lr': lr_init})


        return out


    def forward(self, x, x_norm = None):
         # get density values at the sample locations
        density_features = self.compute_densityfeature(x_norm)
        # density_features = self.compute_densityfeature3d_fast(x)
        # convert density values to density values
        density_features = self.feature2density(density_features)

        values_target = self.get_values_for_coords(self.downsampled_target, x)

        loss = self.loss_fn(density_features, values_target)

        reg_term = torch.tensor(0.0, device=self.device) #TODO: implement regularization
        
        return loss, reg_term

    def feature2density(self, density_features):
        return density_features
        # return F.relu(density_features)


    @torch.no_grad()
    def get_image_reconstruction(self, batch_size=512**2):
        #model.downsample_target to cpu
        self.downsampled_target = self.downsampled_target.cpu()


        grid_size = self.grid_size

        reconstructed_density = torch.zeros(grid_size)  # Move tensor to the appropriate device


        # Create a SimpleSamplerNonRandom with the appropriate dimensions and grid size
        sample_dim = len(grid_size)
        if self.payload > 1:
            sample_dim -= 1
        sampler = SimpleSamplerNonRandom(sample_dim, batch_size, max_value=grid_size[0] - 1)
        num_samples = sampler.total_samples

        # Calculate the number of iterations needed
        num_iterations = num_samples // batch_size + 1

        # Create a tqdm progress bar
        for _ in tqdm(range(num_iterations), desc="Processing Batches in Batched PSNR"):
            batch_indices, batch_indices_norm = sampler.next_batch()
            
            density_features = self.compute_densityfeature(batch_indices_norm.to(self.device))

            # Move batch_indices tensor to the appropriate device
            #batch_indices = batch_indices.to(self.device)

            # Use grid_size to determine the indices of the samples
            if sample_dim == 2:
                reconstructed_density[batch_indices[:, 0], batch_indices[:, 1]] = density_features.detach().cpu()
            elif sample_dim == 3:
                reconstructed_density[batch_indices[:, 0], batch_indices[:, 1], batch_indices[:, 2]] = density_features.detach().cpu()
            
        # back to gpu
        self.downsampled_target = self.downsampled_target.to(self.device)
        
        return reconstructed_density

    @torch.no_grad()
    def upsample(self, iteration):
        #calculate target resolution based on iteration, init_reso and grid_size
        self.current_reso = self.init_reso * 2**(iteration+1)
        #res_target = [self.current_reso for i in range(len(self.grid_size))]
        res_target = [self.current_reso  for i in range(self.dimensions)] 
        if self.payload_position != 'grayscale' and self.payload != 0:
            res_target.append(self.payload)

        self.grid_size = res_target
        downscale_factor = int(self.target.shape[1]/self.current_reso) # works for both grayscale and color images
        self.downsampled_target.cpu()
        self.downsampled_target = self.downsample_target(factor= downscale_factor, grayscale=self.grayscale, dim=self.dimensions, masked_avg_pooling = self.masked_avg_pooling)

        self.upsample_volume_grid(res_target)

        if downscale_factor == 0:
            raise ValueError("Downscale factor is 0. This means that the target image is too small for the current resolution. Try to increase the initial resolution or decrease the number of upsamplings.")

        self.set_compression_variables()


    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
        return density_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        # take density line of gpu
        self.density_line = self.density_line.cpu()
        self.density_line = self.up_sampling_Vector(self.density_line, res_target)
        self.density_line = self.density_line.to(self.device) # test if makes a difference for large tensors
        

    def reconstruct(self):
        # Implementation
        pass


    def set_compression_variables(self):
        self.dtype_sz_bytes = 4 # torch.float32
        self.num_uncompressed_params = np.prod(self.grid_size)
        self.num_compressed_params = sum([np.prod(self.density_line[i].shape) for i in range(len(self.density_line))])
        # if payload > 1 add basis_mat
        if self.payload > 1:
            self.num_compressed_params += np.prod(self.basis_mat.weight.shape)
        #self.num_compressed_params = sum(p.numel() for p in self.torch_params.values())
        self.compression_factor = self.num_uncompressed_params/self.num_compressed_params
        self.sz_uncompressed_gb = self.num_uncompressed_params * self.dtype_sz_bytes / 1024**3 # in GB
        self.sz_compressed_gb = self.num_compressed_params * self.dtype_sz_bytes / 1024**3 # in GB

    def compute_total_variation_loss(self, img, weight=1.0):
        # Implementation
        pass

    def l1_regularization(self):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def l2_regularization(self):
        # Implementation
        pass

    def apply_activation(self, x):
        # Implementation
        pass