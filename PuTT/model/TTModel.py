# Based on TensoRF code from https://github.com/apchenstu/TensoRF
# Modifications and/or extensions have been made for specific purposes in this project.



import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tn_utils import *
from utils import SimpleSamplerNonRandom

from tqdm import tqdm

from model.BaseTNModel import BaseTNModel
from opt_einsum import contract, contract_path

class TTModel(BaseTNModel):
    def __init__(self, target, init_reso, max_rank=50, dtype='float32', loss_fn_str="L2", use_TTNF_sampling=False, payload=0, payload_position='first_core', canonization="first", activation="None", compression_alg="compress_all", regularization_type="TV", dimensions =3, regularization_weight=0.0, noisy_target = None, device = 'cpu', masked_avg_pooling=False, is_tensor_ring = True, channel_rank = None):
        super().__init__(target, init_reso, max_rank, dtype, loss_fn_str, use_TTNF_sampling, payload, payload_position, canonization, activation, compression_alg, regularization_type, dimensions, noisy_target, device, masked_avg_pooling)
        # Additional initialization specific to TNmodeCP
        self.model = "TT"
        self.scale_init = 0.1
        self.is_tensor_ring = is_tensor_ring
        self.channel_rank = channel_rank
        self.vecMode = list(range(dimensions))[::-1]
        self.density_n_comp = self.initialize_density_n_comp(dimensions, max_rank)
        self.payload = max(1, payload) if is_tensor_ring else payload  # Ensure payload is at least 1 for tensor ring
        self.dimensions = dimensions
        self.regularization_weight = regularization_weight
        self.use_TTNF_sampling = use_TTNF_sampling
        
        self.rank_upsampling_masking = True
        self.mask_rank = 1000
        
        self.init_tn()
        self.set_compression_variables()
        
        self.optimize_contraction_expression_density()
        
        
        
        
    def create_rank_upsampling_mask(self):
        mask_components = []
        
        # componets have shape (r1, n, r2)
        
        for i, comp in enumerate(self.components):
            mask = torch.ones_like(comp)
            # from mask rank set to 0
            mask[self.mask_rank:, :, self.mask_rank:] = 0
            mask_components.append(mask)
            
        self.mask_components = mask_components

    def initialize_density_n_comp(self, dimensions, max_rank):
        density_n_comp = [max_rank for _ in range(dimensions)]
        if self.is_tensor_ring:
            density_n_comp.append(self.channel_rank)
        return density_n_comp
    
    def init_tn(self):
        init_method = self.init_one_svd_3d if self.dimensions == 3 else self.init_one_svd_2d
        self.components = init_method(self.density_n_comp, self.grid_size, self.scale_init, self.device)
        if self.dimensions == 2:
            self.x_component, self.y_component, self.channel_component = self.components
        elif self.dimensions == 3:
            self.x_component, self.y_component, self.z_component, self.channel_component = self.components
            
        print("x_component", self.x_component.shape)
        print("y_component", self.y_component.shape)
        if self.channel_component is not None:
            print("channel_component", self.channel_component.shape)
            
        if self.rank_upsampling_masking:
            self.create_rank_upsampling_mask()
            

    def init_one_svd_2d(self, ranks, grid_size, scale, device):
        # Tensor component initialization for 2D case
        if self.payload_position == 'grayscale' or self.payload < 2:
            x_component = torch.nn.Parameter(scale * torch.randn((ranks[-1], grid_size[0], ranks[0]))).to(device)
            y_component = torch.nn.Parameter(scale * torch.randn((ranks[0], grid_size[1], ranks[-1]))).to(device)
            channel_component = None
        elif self.is_tensor_ring:
            x_component = torch.nn.Parameter(scale * torch.randn((ranks[-1], grid_size[0], ranks[0]))).to(device)
            y_component = torch.nn.Parameter(scale * torch.randn(( ranks[0], grid_size[0], ranks[-1]))).to(device)
            channel_component = torch.nn.Parameter(scale * torch.randn((ranks[-1], self.payload if self.payload > 1 else 1, ranks[-1]))).to(device)
        else:
            x_component = torch.nn.Parameter(scale * torch.randn((1, grid_size[0], ranks[0]))).to(device)
            y_component = torch.nn.Parameter(scale * torch.randn((ranks[0],grid_size[1], self.payload if self.payload > 1 else 1))).to(device)
            channel_component = None
        
        return x_component, y_component, channel_component

    def init_one_svd_3d(self, ranks, grid_size, scale, device):
        # Tensor component initialization for 3D case
        if (self.payload_position == 'grayscale' or self.payload < 2) and not self.is_tensor_ring:
            x_component = torch.nn.Parameter(scale * torch.randn((1, grid_size[0], ranks[0]))).to(device)
            y_component = torch.nn.Parameter(scale * torch.randn((ranks[0], grid_size[1], ranks[1]))).to(device)
            z_component = torch.nn.Parameter(scale * torch.randn((ranks[1], grid_size[2], self.payload if self.payload > 1 else 1))).to(device)
            channel_component = None
        elif self.is_tensor_ring:
            x_component = torch.nn.Parameter(scale * torch.randn((ranks[-1], grid_size[0], ranks[0]))).to(device)
            y_component = torch.nn.Parameter(scale * torch.randn((ranks[0], grid_size[1], ranks[1]))).to(device)
            z_component = torch.nn.Parameter(scale * torch.randn((ranks[0], grid_size[2], ranks[-1]))).to(device)
            channel_component = torch.nn.Parameter(scale * torch.randn((ranks[-1], self.payload if self.payload > 1 else 1, ranks[-1]))).to(device)
        else:
            x_component = torch.nn.Parameter(scale * torch.randn((1, grid_size[0], ranks[0]))).to(device)
            y_component = torch.nn.Parameter(scale * torch.randn((ranks[0], grid_size[1], ranks[0]))).to(device)
            z_component = torch.nn.Parameter(scale * torch.randn((ranks[0], grid_size[2], ranks[1]))).to(device)
            channel_component = None

        return x_component, y_component, z_component, channel_component

    def compute_densityfeature(self, samples):
        compute_method = self.compute_densityfeature3d if self.dimensions == 3 else self.compute_densityfeature2d
        return compute_method(samples)      
        
    
    def compute_densityfeature2d(self, xy_sampled):

        if self.use_TTNF_sampling:
            input_ = [self.x_component, self.y_component, self.channel_component] if self.is_tensor_ring else [self.x_component, self.y_component]
            coords = coords_to_v2_tensor_ring(xy_sampled)
            last_core_is_payload = True if self.is_tensor_ring else False
            values_recon = sample_intcoord_tt_ring_v2(input_, coords, last_core_is_payload=last_core_is_payload, checks=True)    
            return values_recon
        
        xy_features = self.contract_tensor_network()
        sampled_density_values = self.get_values_for_coords(xy_features, xy_sampled)
        return sampled_density_values
        
    def compute_densityfeature3d(self, xyz_sampled):
        # Compute the intermediate feature by contracting x_component and y_component
        xyz_features = self.contract_tensor_network()
        sampled_values = self.get_values_for_coords(xyz_features, xyz_sampled)
        return sampled_values
    
    def contract_tensor_network(self):
        
        if self.mask_rank != self.max_rank and self.rank_upsampling_masking:
            return self.masked_contract_tensor_network()
        
        return torch.einsum(self.einsum_str, *self.components).squeeze() # non-optimized version
        
        #return contract(self.einsum_str, *self.components, optimize=self.optimized_einsum_path).squeeze()
    
    def masked_contract_tensor_network(self):
        components_masked = [comp * mask for comp, mask in zip(self.components, self.mask_components)]
        return torch.einsum(self.einsum_str, *components_masked).squeeze()
    
    def optimize_contraction_expression_density(self):
        if self.dimensions == 2:
            if self.payload_position == 'grayscale' or self.payload < 2:
                einsum_str = 'ixj,jyi->xy' 
                components = (self.x_component, self.y_component) 
            else:
                einsum_str = 'ixj,jyp,icp->xyc' if self.is_tensor_ring else 'ixj,jyc->xyc'
                components = (self.x_component, self.y_component, self.channel_component) if self.is_tensor_ring else (self.x_component, self.y_component)
        else:
            einsum_str = 'ixj,jyk,kzp,pci->xyzc' if self.is_tensor_ring or self.payload > 1 else 'ixj,jyk,kzq->xyz'
            components = (self.x_component, self.y_component, self.z_component, self.channel_component) if self.is_tensor_ring or self.payload > 1 else (self.x_component, self.y_component, self.z_component)
            
        self.optimized_einsum_path = self.optimize_contraction_expression(components, einsum_str)
        self.components = components
        self.einsum_str = einsum_str
    
    def optimize_contraction_expression(self, components, einsum_str):
        # Implementation
        #path_info = contract_path(einsum_str, *components,æ optimize='optimal')
        path_info = contract_path(einsum_str, *components)#, optimize='optimal')
        self.path_info = path_info
        return path_info[0]

    def get_optparam_groups(self, lr_init=0.02, lr_init_network=0.001):
        # Start with an empty list for parameters
        params = []

        # Add common components
        params.extend([self.x_component, self.y_component])

        # Add dimension-specific and configuration-specific components
        if self.dimensions == 3:
            params.append(self.z_component)

        if self.is_tensor_ring and self.payload_position != 'grayscale' or (self.dimensions == 3 and self.payload > 1):
            params.append(self.channel_component)

        # Create the gradient variables list with specified learning rates
        grad_vars = [{'params': params, 'lr': lr_init}]

        return grad_vars


    def feature2density(self, density_features):
        return density_features
        # return F.relu(density_features)


    @torch.no_grad()
    def get_image_reconstruction(self, batch_size=None):
        
        return self.contract_tensor_network()
        

    @torch.no_grad()
    def up_sampling_TT(self, x_component, y_component, z_component = None, res_target = [32, 32, 32]):
        # Function to perform up-sampling on tensor components to match target resolutions
        
        def upsample_component_trilinear(component, target_size, vec_mode):
            # Upsample a single component
            # Unsqueeze to add a batch dimension and permute to rearrange dimensions for interpolation
            if len(component.shape) == 4:
                component = component.unsqueeze(0)
                component = component.permute(0, 2, 1, 3, 4)
            else:
                # component = component.unsqueeze(0).permute(0, 2, 1, 3).unsqueeze(-1)
                component = component.unsqueeze(0).unsqueeze(-1)
            
            # Perform interpolation and then permute back to original dimension arrangement
            component_interpolated = F.interpolate(
                component, 
                size=(target_size[vec_mode], component.shape[-2],1),  # Only modify the target dimension
                mode='trilinear', 
                align_corners=True
            )
            
            component_interpolated = component_interpolated.squeeze(0) # Remove the added batch dimension after interpolation

            # Move the upsampled component to the specified device
            return torch.nn.Parameter(component_interpolated).to(self.device)

        # Apply up-sampling to each component
        x_component = upsample_component_trilinear(x_component, res_target, self.vecMode[0])
        y_component = upsample_component_trilinear(y_component, res_target, self.vecMode[1])
        
        if 1 not in self.density_n_comp:
            x_component.squeeze_(-1)
            y_component.squeeze_(-1)
        if self.dimensions == 2:
            return x_component, y_component
        
        z_component = upsample_component_trilinear(z_component, res_target, self.vecMode[2]).squeeze_(-1)
        
        
        return x_component, y_component, z_component
    
        

    @torch.no_grad()
    def upsample(self, iteration):
        # take downsampled target, density_plane, density_line of gpu
        self.downsampled_target = self.downsampled_target.cpu()
        # VM can be heavy on GPU memory, so we move the tensors to CPU
        self.x_component = self.x_component.cpu()
        self.y_component = self.y_component.cpu()
        if self.dimensions == 3:
            self.z_component = self.z_component.cpu()


        #calculate target resolution based on iteration, init_reso and grid_size
        self.current_reso = self.init_reso * 2**(iteration+1)
        #res_target = [self.current_reso for i in range(len(self.grid_size))]
        res_target = [self.current_reso  for i in range(self.dimensions)] 
        if self.payload_position != 'grayscale' or self.payload > 1:
            res_target.append(self.payload)

        self.grid_size = res_target
        downscale_factor = int(self.target.shape[1]/self.current_reso) # works for both grayscale and color images
        self.downsampled_target = self.downsample_target(factor= downscale_factor, grayscale=self.grayscale, dim=self.dimensions, masked_avg_pooling = self.masked_avg_pooling)

        if self.dimensions == 2:
            self.x_component, self.y_component = self.up_sampling_TT(self.x_component, self.y_component, None, res_target)
        elif self.dimensions == 3:
            self.x_component, self.y_component, self.z_component = self.up_sampling_TT(self.x_component, self.y_component, self.z_component, res_target)

        if downscale_factor == 0:
            raise ValueError("Downscale factor is 0. This means that the target image is too small for the current resolution. Try to increase the initial resolution or decrease the number of upsamplings.")

        self.optimize_contraction_expression_density()
        
        print("self.x_component.shape", self.x_component.shape)
        print("self.y_component.shape", self.y_component.shape)
        if self.dimensions == 3:
            print("self.z_component.shape", self.z_component.shape)
        self.set_compression_variables()        
        
        components = [self.x_component, self.y_component]
        if self.dimensions == 3:
            components.append(self.z_component)
        if self.payload_position != 'grayscale' or self.payload > 1:
            components.append(self.channel_component)
        self.create_rank_upsampling_mask()
        
        
    def upsample_ranks(self, padding=2):
        
        components = [self.x_component, self.y_component]
        if self.dimensions == 3:
            components.append(self.z_component)
        if self.payload_position != 'grayscale' or self.payload > 1:
            components.append(self.channel_component)
        
        components = self.upsample_rank_componets(components, padding)
        if self.dimensions == 3:
            if self.payload_position != 'grayscale' or self.payload > 1:
                x_component, y_component, z_component, channel_component = components
            else:
                x_component, y_component, z_component = components  
        else:
            if self.payload_position != 'grayscale' or self.payload > 1:
                x_component, y_component, channel_component = components
            else:
                x_component, y_component = components
            
        self.x_component = torch.nn.Parameter(x_component).to(self.device)
        self.y_component = torch.nn.Parameter(y_component).to(self.device)
        if self.dimensions == 3:
            self.z_component = torch.nn.Parameter(z_component).to(self.device)
        if self.payload_position != 'grayscale' or self.payload > 1:
            self.channel_component = torch.nn.Parameter(channel_component).to(self.device)
        self.optimize_contraction_expression_density()
        self.set_compression_variables()
        
        
        
        
    def upsample_rank_componets(self, components, padding=2):
        # for each component, upsample the ranks with a   addtion of 2 - padding with 0s
        # Implementation
        upsampled_components = []
        for comp in components:
            comp = comp.data
            # comp is expected to be a tensor of shape (r1, n, r2)
            # Pad r1 and r2 dimensions with 'padding' zeros. The middle dimension 'n' remains unchanged.
            # The padding sequence is (left, right, top, bottom, front, back) for the last two dimensions.
            pad_sequence = (padding // 2, padding // 2, 0, 0, padding // 2, padding // 2)
            upsampled_comp = F.pad(comp, pad_sequence, "constant", 0)  # Padding with 0s
            upsampled_components.append(upsampled_comp)
            
        return upsampled_components
    def reconstruct(self):
        # Implementation
        pass


    def set_compression_variables(self):
        self.dtype_sz_bytes = 4 # torch.float32
        self.num_uncompressed_params = np.prod(self.grid_size)
        print("num_uncompressed_params",self.num_uncompressed_params)
        # add the number of parameters in the x_component, y_component, z_component tensors
        if self.dimensions == 2:
            self.num_compressed_params = sum([np.prod(self.x_component.shape), np.prod(self.y_component.shape)])
        elif self.dimensions == 3:
            self.num_compressed_params = sum([np.prod(self.x_component.shape), np.prod(self.y_component.shape), np.prod(self.z_component.shape)])
            
        if self.is_tensor_ring and self.payload_position != 'grayscale':
            self.num_compressed_params += np.prod(self.channel_component.shape)
            
        #self.num_compressed_params = sum(p.numel() for p in self.torch_params.values())
        self.compression_factor = self.num_uncompressed_params/self.num_compressed_params
        self.sz_uncompressed_gb = self.num_uncompressed_params * self.dtype_sz_bytes / 1024**3 # in GB
        self.sz_compressed_gb = self.num_compressed_params * self.dtype_sz_bytes / 1024**3 # in GB

    
    def compute_total_variation_loss(self):
        loss = 0
        if self.dimensions == 2:
            components = [self.x_component, self.y_component]
        elif self.dimensions == 3:
            components = [self.x_component, self.y_component, self.z_component]
        for comp in components:
            # each is tensor of shape (r1, n, r2) - do difference along the n dimension
            loss += torch.sum(torch.abs(comp[:, 1:, :] - comp[:, :-1, :]))
        return loss / self.num_uncompressed_params
        
    def l1_regularization(self):
        total = 0
        components =[self.x_component, self.y_component]
        if self.dimensions == 3:
            components.append(self.z_component)
        if self.is_tensor_ring and self.payload_position != 'grayscale':
            components.append(self.channel_component)
        for comp in components:
            total = total + torch.mean(torch.abs(comp))
            
        return total 

    def l2_regularization(self):
        # Implementation
        pass

    def apply_activation(self, x):
        # Implementation
        pass
    
    def compute_regularization_term(self):
        """
        Computes the regularization term for the tensor network.

        Parameters:
        - tn: the tensor network.

        Returns:
        The regularization term.
        """
        if self.regularization_type == "TV":
            reg_term = self.compute_total_variation_loss()
        elif self.regularization_type == "L1":
            reg_term = self.l1_regularization()
        elif self.regularization_type == "L2":
            reg_term = self.l2_regularization()
        else: 
            reg_term = 0
        
        return reg_term
    
    
    def forward(self, x, x_norm = None):
         # get density values at the sample locations
        density_values = self.compute_densityfeature(x)

        values_target = self.get_values_for_coords(self.downsampled_target, x)

        loss = self.loss_fn(density_values, values_target)
        
        #reg_term = torch.tensor(0.0, device=self.device) # TODO: implement regularization
        reg_term = self.compute_regularization_term()

        return loss, reg_term
    