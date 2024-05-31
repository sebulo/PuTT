# Based on TensoRF code from https://github.com/apchenstu/TensoRF
# Modifications and/or extensions have been made for specific purposes in this project.

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tensorly import tucker_to_tensor
from tn_utils import *
from utils import SimpleSamplerNonRandom

from tqdm import tqdm

from model.BaseTNModel import BaseTNModel

class TuckerModel(BaseTNModel):
    def __init__(self, target, init_reso, max_rank=50, dtype='float32', loss_fn_str="L2", use_TTNF_sampling=False, payload=0, payload_position='first_core', canonization="first", activation="None", compression_alg="compress_all", regularization_type="TV", dimensions =2, regularization_weight=0.0, noisy_target = None, device = 'cpu', masked_avg_pooling=False):
        super().__init__(target, init_reso, max_rank, dtype, loss_fn_str, use_TTNF_sampling, payload, payload_position, canonization, activation, compression_alg, regularization_type, dimensions, noisy_target, device, masked_avg_pooling)
        # Additional initialization specific to TNmodeCP
        self.model = "Tucker"

        self.scale_init = 0.2
        self.vecMode = list(range(dimensions))[::-1]
        self.density_n_comp = [max_rank for i in range(dimensions)]

        self.init_tn()
        self.set_compression_variables()


    def init_tn(self):
        factors = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            
            factors.append(torch.nn.Parameter(
            self.scale_init * torch.randn((self.grid_size[vec_id], self.density_n_comp[i]), requires_grad=True)))
            
            core = torch.nn.Parameter(
                self.scale_init * torch.randn(*self.density_n_comp, requires_grad=True))

        self.factors = torch.nn.ParameterList(factors).to(self.device)
        self.core = torch.nn.ParameterList([core]).to(self.device)
        if self.payload > 1:
            self.basis_mat = torch.nn.Linear(self.density_n_comp[0], self.payload, bias=False).to(self.device)

    def compute_densityfeature(self, samples):
        if self.dimensions ==2:
            return self.compute_densityfeature2d(samples)
        elif self.dimensions ==3:
            return self.compute_densityfeature3d(samples)
        else:
            raise NotImplementedError


    def compute_densityfeature2d(self, xy_sampled):
        index_factor_x = xy_sampled[..., self.vecMode[0]]
        index_factor_y = xy_sampled[..., self.vecMode[1]]

        temp = torch.matmul(self.factors[0][index_factor_x], self.core[0])

        if self.payload > 1:
            new_temp = temp * self.factors[1][index_factor_y] 
            sigma_feature =  torch.matmul(new_temp, self.basis_mat.weight.T)
        else:
            sigma_feature = torch.sum(temp * self.factors[1][index_factor_y], dim=1)
        return sigma_feature
    
    def compute_densityfeature3d(self, xyz_sampled):
        tucker_tensor = tucker_to_tensor((self.core[0], [self.factors[0], self.factors[1], self.factors[2]]))
        res = tucker_tensor[xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]]
        return res
        


    # Look at this again TODO
    def get_optparam_groups(self, lr_init = 0.02):
        out = []
        if self.payload > 1:
            out.append({'params': self.basis_mat.weight, 'lr': lr_init})
        for i in range(len(self.vecMode)):
            out.append({'params': self.factors[i], 'lr': lr_init})
        out.append({'params': self.core, 'lr': lr_init})
        return out
    
        

    def forward(self, x, x_norm = None):
         # get density values at the sample locations
        density_features = self.compute_densityfeature(x)
        # density_features = self.compute_densityfeature(x_norm)
        # convert density values to density values
        density_features = self.feature2density(density_features)

        values_target = self.get_values_for_coords(self.downsampled_target, x)

        loss = self.loss_fn(density_features, values_target)

        return loss

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
            batch_indices, _ = sampler.next_batch()
            
            density_features = self.compute_densityfeature(batch_indices.to(self.device))

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
        self.downsampled_target = self.downsample_target(factor= downscale_factor, grayscale=self.grayscale, dim=self.dimensions, masked_avg_pooling = self.masked_avg_pooling)

        self.upsample_volume_grid(res_target)

        if downscale_factor == 0:
            raise ValueError("Downscale factor is 0. This means that the target image is too small for the current resolution. Try to increase the initial resolution or decrease the number of upsamplings.")

        self.set_compression_variables()


    @torch.no_grad()
    def up_sampling_tucker(self, factors, core, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            # Expand the factor tensor to introduce a batch dimension
            expanded_factor = factors[i].unsqueeze(0).unsqueeze(0)
            # Interpolate
            expanded_factor = torch.nn.Parameter(
                F.interpolate(expanded_factor, size=(res_target[vec_id], self.density_n_comp[i]), mode='bilinear', align_corners=True))
            
            factors[i] = expanded_factor.squeeze(0).squeeze(0)
        
        return factors, core
    

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.factors = self.factors.cpu()
        self.core = self.core.cpu()
        self.factors, self.core = self.up_sampling_tucker(self.factors, self.core, res_target)
        self.core.to(self.device)
        self.factors.to(self.device)

    def reconstruct(self):
        # Implementation
        core = self.core[0].cpu().detach().numpy()
        factors = [self.factors[i].cpu().detach().numpy() for i in range(len(self.factors))]
        tucker_tensor = tucker_to_tensor((core[0], [factors[0], factors[1], factors[2]]))
        return tucker_tensor


    def set_compression_variables(self):
        self.dtype_sz_bytes = 4 # torch.float32
        self.num_uncompressed_params = np.prod(self.grid_size)
        self.num_compressed_params = sum([np.prod(self.factors[i].shape) for i in range(len(self.factors))])
        self.num_compressed_params += sum( [np.prod(self.core[i].shape) for i in range(len(self.core))])
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
        for idx in range(len(self.factors )):
            total = total + torch.mean(torch.abs(self.factors[idx]))

        total = total + torch.mean(torch.abs(self.core))
        return total

    def l2_regularization(self):
        # Implementation
        pass

    def apply_activation(self, x):
        # Implementation
        pass