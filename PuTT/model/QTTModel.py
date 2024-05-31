# Based on TT-NF code from https://github.com/toshas/ttnf
# Modifications and/or extensions have been made for specific purposes in this project.

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tn_utils import *
from utils import  SimpleSamplerNonRandom
# import quimb
import quimb

from model.BaseTNModel import BaseTNModel

class QTTModel(BaseTNModel):
    def __init__(self, target, init_reso, max_rank=256, dtype='float32', loss_fn_str="L2", use_TTNF_sampling=False, payload=0, payload_position='first_core', canonization="first", activation="None", compression_alg="compress_all", regularization_type="TV", dimensions =2, regularization_weight = 0.0, noisy_target = None, device = 'cpu',  masked_avg_pooling = False, sigma_init=0):
        """
        Initializes the QTTModel object.

        Parameters:
        - target: the target tensor to model.
        - init_reso: the initial side length of the tensor.
        - max_rank: maximum rank for tensor decompositions.
        - dtype: data type for computations (default 'float32').
        - loss_fn_str: loss function to be used (e.g., "L2").
        - use_TTNF_sampling: whether to use TTNF V2 sampling - See TTNF Obukhov et. al. 2023.
        - payload: additional payload dimensions.
        - payload_position: the position of the payload in the tensor network - either 'first_core' or 'grayscale': No payload (for grayscale images)
        - canonization: method for canonization in tensor network.
        - activation: activation function to be used (e.g., "None", "relu").
        - compression_alg: algorithm for tensor compression - either 'compress_all' (TT-SVD) or 'compress_tree'.
        - regularization_type: type of regularization (e.g., "TV" for total variation).
        - dimensions: the number of dimensions of the input - e.g. 2 for 2D or and 3 for 3D structures
        - regularization_weight: weight of the regularization term.
        - noisy_target: noisy version of the target tensor - for experiments with noisy or incomplete data
        - device: computation device (e.g., 'cpu', 'cuda').
        - masked_avg_pooling: whether to use masked average pooling - used for incomplete data experiments
        """
        
        super().__init__(target, init_reso, max_rank, dtype, loss_fn_str, use_TTNF_sampling, payload, payload_position, canonization, activation, compression_alg, regularization_type, dimensions, noisy_target, device, masked_avg_pooling, sigma_init)
        self.model = "QTT"
        self.canonization = canonization
        self.compression_alg = compression_alg
        self.regularization_type = regularization_type
        self.activation = activation

        self.shape_source = None
        self.shape_factors = None
        self.factor_target_to_source = None
        self.dim_grid_log2 = int(np.log2(init_reso))


        self.inds = 'k'
        self.use_TTNF_sampling = use_TTNF_sampling

        self.regularization_weight = regularization_weight
        
        self.mask_rank = max_rank

        self.init_tn()
        
        if self.mask_rank < max_rank:
            self.create_rank_upsampling_mask()
        else:
            self.masked_components = None

        self.dim = len(self.shape_source)
        self.iteration = 0

        # extract the raw arrays and a skeleton of the TN
        self.set_torch_params()

        self.set_compression_variables()

        # get equation for contraction
        self.contraction_expression, self.path_info, self.symbol_info = self.get_contraction_expression()
        
    def get_contraction_expression(self):
        """
        Computes the contraction expression for the tensor network using opt_einsum.

        Returns:
        A tuple containing the contraction expression, path information, and symbol information.
        """
        
        output_inds = self.tn.outer_inds()
        if self.payload_position != 'grayscale':
            output_inds = [ind for ind in output_inds if ind != 'payload'] # remove string "payload" from output inds 
            output_inds = ['payload'] + output_inds # add to front of output inds
        backend = 'torch'
        optimze = 'dp'
        path_info =self.tn.contract( output_inds=output_inds, get='path-info', backend=backend, optimize=optimze)
        self.update_contraction_costs(path_info)
        symbol_info =self.tn.contract( output_inds=output_inds, get='symbol-map', backend = backend, optimize=optimze)
        contraction_expression = self.tn.contract(output_inds=output_inds, get='expression', backend=backend, optimize=optimze)
        
        return contraction_expression, path_info, symbol_info

    def init_tn(self):
        """
        Initializes the tensor network (TN) for the model.
        """
        # Create the initial QTTNF
        self.tn, self.shape_source, self.shape_target, self.shape_factors, _, self.factor_target_to_source = get_qtt_TTNF(self.current_reso, self.max_rank, dim=self.dimensions, payload_dim=self.payload, payload_position=self.payload_position, compression_alg=self.compression_alg, canonization=self.canonization, sigma_init=self.sigma_init)      
        print("Initialized tn,", self.tn)
        
    
    def create_rank_upsampling_mask(self):
        # take self.tn and create a mask of ones for each core 
        masked_components = []
        for i,c in enumerate(self.tn.tensors):
            mask = torch.ones_like(c.data)
            # all cores have form (r1, n, r2) except the last one which is (r1, n) and the first one which is (Payload n, r2)
            # set to zeros where exceeding the mask rank
            if i == 0:
                if len(mask.shape) == 2:
                    mask[:,self.mask_rank:] = 0
                else:
                    mask[ :, :,self.mask_rank:] = 0
            elif i == len(self.tn.tensors)-1:
                mask[ self.mask_rank:, :] = 0
            else:
                mask[self.mask_rank:, :, self.mask_rank:] = 0
        
            masked_components.append(mask)
            
        self.masked_components = masked_components
    
    
        

    def set_torch_params(self, core_indices_to_exclude = []):
        """
        Retrieves the parameters of the tensor network as a PyTorch ParameterDict.

        Parameters:
        - core_indices_to_exclude: indices of cores to exclude from training (making them non-trainable).

        Returns:
        A PyTorch ParameterDict containing the parameters.
        """
        
        params, self.skeleton = qtn.pack(self.tn)

        self.torch_params = torch.nn.ParameterDict({
            # torch requires strings as keys
            str(i): torch.nn.Parameter(initial)
            for i, initial in params.items()
        })
        # if core_indices_to_exclude length is not 0, remove the corresponding cores from the torch_params making them non-trainable
        if len(core_indices_to_exclude) != 0:
            for i in core_indices_to_exclude:
                self.torch_params[str(i)].requires_grad = False

        self.tn = self.reconstruct() # Very important!
        
        return self.torch_params

    def get_optparam_groups(self, lr_init = 0.02):
        """
        Groups the optimization parameters.

        Parameters:
        - lr_init: initial learning rate for the optimization.

        Returns:
        A list of dictionaries containing parameters and their learning rates.
        """
        
        out = []
        out += [
            {'params': self.torch_params.values(), 'lr': lr_init},
        ]
        return out

    
    def save_tn(self, base_path):
        """
        Saves the tensor network's parameters and skeleton to disk.

        Parameters:
        - base_path: base path for saving the files.
        """
        
        """ Save the current TN parameters and skeleton to disk with distinct names. """
        params_path = base_path + "_params.pth"  # Parameters file
        torch.save(self.torch_params, params_path)
        # quimb.save_to_disk(self.skeleton, skeleton_path)

    def load_tn(self, base_path):
        """
        Loads the tensor network's parameters and skeleton from disk.

        Parameters:
        - base_path: base path for loading the files.
        """
        
        params_path = base_path + "_params.pth"  # Parameters file
        skeleton_path = base_path + "_skeleton"  # Skeleton file

        self.torch_params = torch.load(params_path)
        self.skeleton = quimb.load_from_disk(skeleton_path)
        params = {int(i): p for i, p in self.torch_params.items()}
        self.tn = qtn.unpack(params, self.skeleton)
        self.set_compression_variables()
        self.update_contraction_costs()
        self.num_trainable_params = sum(p.numel() for p in self.torch_params.values())
        self.contraction_expression, self.path_info, self.symbol_info = self.get_contraction_expression()
                
    @torch.no_grad()
    def upsample(self, iteration):
        """
        Upsamples the tensor network.

        Parameters:
        - iteration: the current iteration number - this is used to keep track of the indices in the tensor network - there is a difference between iterations in first and subsequent upsamplings.
        """
        self.tn = self.reconstruct() # Very important!
        
        # New - helps stabilize upsampling
        norm = self.tn.norm()
        self.tn /= self.tn.norm()
        
        self.current_reso = self.current_reso * 2
        self.dim_grid_log2 = self.dim_grid_log2 + 1
        downscale_factor = int(self.target.shape[1]/self.current_reso) # works for both grayscale and color images
        if downscale_factor == 0:
            raise ValueError("Downscale factor is 0. This means that the target image is too small for the current resolution. Try to increase the initial resolution or decrease the number of upsamplings.")
        self.downsampled_target = self.downsampled_target.cpu()
        self.downsampled_target = self.downsample_target(factor = downscale_factor, grayscale = self.grayscale, dim = self.dimensions, masked_avg_pooling = self.masked_avg_pooling)
            
        self.tn, self.shape_source, self.shape_factors, self.factor_target_to_source = prolongate_qtt(self.tn, dim=self.dimensions, ranks_tt= self.max_rank, payload_position = self.payload_position, compression_alg = self.compression_alg, canonization = self.canonization)

        # New - helps stabilize upsampling
        self.tn /= self.tn.norm()
        self.tn = self.tn * norm
        
        self.iteration += 1 # update to keep tracik of inds in prolongation

        self.update_tn_params()
        if self.max_rank > self.mask_rank:
            self.create_rank_upsampling_mask()
        
        
    def update_tn_params(self):
        self.set_torch_params()
        self.set_compression_variables()
        self.update_contraction_costs()
        self.num_trainable_params = sum(p.numel() for p in self.torch_params.values())
        self.contraction_expression, self.path_info, self.symbol_info = self.get_contraction_expression()

    def reconstruct(self):
        """
        Reconstructs the tensor network as a quimb tensor network.

        Returns:
        The reconstructed tensor network.
        """
        # convert back to original int key format
        params = {int(i): p for i, p in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        tn = qtn.unpack(params, self.skeleton)

        return tn

    def get_image_reconstruction(self):
        """
        Reconstructs the image from the tensor network.

        Parameters:
        - tn: the tensor network to reconstruct the image from (if None, uses the current model's network).

        Returns:
        The reconstructed image.
        """
        data = qtt_to_tensor(self.tn, self.shape_source, self.shape_factors, self.factor_target_to_source,
                            inds=self.inds, payload_position=self.payload_position, grayscale= self.grayscale,
                            expression=self.contraction_expression, payload=self.payload, masked_components = self.masked_components)
                            # expression=None)
        return data
    
    def set_compression_variables(self):
        """
        Sets the compression variables for the tensor network.
        """
        
        if self.dtype == torch.float32:
            self.dtype_sz_bytes = 4
        elif self.dtype == torch.float64:
            self.dtype_sz_bytes = 8

        self.num_uncompressed_params = np.prod(self.shape_source)
        if self.payload != 0:
            self.num_uncompressed_params = self.num_uncompressed_params * self.payload
        self.num_compressed_params = sum(p.numel() for p in self.torch_params.values())
        
        self.sz_uncompressed_gb = self.num_uncompressed_params * self.dtype_sz_bytes / (1024 ** 3) 
        self.sz_compressed_gb = self.num_compressed_params * self.dtype_sz_bytes / (1024 ** 3)
        self.compression_factor = self.num_uncompressed_params / self.num_compressed_params
    
    def update_contraction_costs(self, contraction_info = None):
        """
        Updates the cost of contraction variable for the tensor network.

        Parameters:
        - contraction_info: the contraction information to use (if None, computes it from the current network).
        """
        if contraction_info is None:
            contraction_info = self.tn.contraction_info()
        self.flops = contraction_info.opt_cost
        self.largest_intermediate = contraction_info.largest_intermediate
        self.flops_per_iter = self.flops_per_iter + [ self.flops]
        self.largest_intermediate_per_iter = self.largest_intermediate_per_iter + [ self.largest_intermediate]
    
    def l1_regularization(self, tn):
        """
        Computes the L1 regularization term for the tensor network.

        Parameters:
        - tn: the tensor network.

        Returns:
        The L1 regularization term.
        """
        total = 0
        for i in range(len(tn.tensors)):
            total += torch.mean(torch.abs(tn.tensors[i].data))
            
        return total 

    def l2_regularization(self, tn):
        """
        Computes the L2 regularization term for the tensor network.

        Parameters:
        - tn: the tensor network.

        Returns:
        The L2 regularization term.
        """
        total = 0
        for i in range(len(tn.tensors)):
            total += torch.mean(torch.pow(tn.tensors[i].data, 2))
            
        return total 
        

    def compute_total_variation_loss(self, tn):
        """
        Computes the total variation loss for the tensor network.

        Parameters:
        - tn: the tensor network with a number of cores len(tn.tensors)

        Returns:
        The total variation loss.
        """
        total_variation_loss = 0
        for tensor in tn.tensors:
            # Ensure tensor has three dimensions
            tensor = tensor.data
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(0)

            # Compute the squared differences in the horizontal direction
            horizontal_diff = torch.pow(tensor[:, :, 1:] - tensor[:, :, :-1], 2)

            # Compute the squared differences in the vertical direction
            vertical_diff = torch.pow(tensor[:, 1:, :] - tensor[:, :-1, :], 2)

            # Sum up the horizontal and vertical differences
            total_variation_loss += torch.sum(horizontal_diff) + torch.sum(vertical_diff)

        # Normalize by the number of paramters
        return total_variation_loss /self.num_compressed_params

    def apply_activation(self, x):
        """
        Applies the specified activation function to the input tensor.

        Parameters:
        - x: the input tensor.

        Returns:
        The tensor after applying the activation function.
        """
        if self.activation == "relu":
            return torch.nn.functional.relu(x)
        elif self.activation == "softplus":
            return torch.nn.functional.softplus(x)
        else:
            return x
        
    @torch.no_grad()
    def batched_qtt(self, compute_reconstruction=False, target = None):
        """
        Performs batched tensor network contraction for large grids that are prohibitively large to contract at once.

        Parameters:
        - compute_reconstruction: whether to compute and return the reconstruction.
        - target: the target tensor to model (if None, uses the model's target).

        Returns:
        The PSNR value and the reconstructed object (if compute_reconstruction is True).
        """

        self.downsampled_target = self.downsampled_target.cpu()

        batch_size = 512**2
        grid_size = [self.current_reso for _ in range(self.dimensions)]
        num_batches = int(np.prod(grid_size) / batch_size)

        # Create a SimpleSamplerNonRandom with the appropriate dimensions and grid size
        sampler = SimpleSamplerNonRandom(self.dimensions, batch_size, max_value=grid_size[0] - 1)

        count = 0
        acc_loss = 0

        if target is None:
            target = self.target

        if target.shape[0] == 1:
            target = target.squeeze(0)
            if self.downsampled_target.shape[-1] == 1:
                target = target.unsqueeze(-1)

        if compute_reconstruction:
            reconstructed_object = torch.zeros(target.shape)
        else :
            reconstructed_object = None
        

        for _ in tqdm(range(num_batches), desc="Processing Batches in Batched QTT"):
            batch_indices, _ = sampler.next_batch()
            values_recon = self.get_reconstructed_values(batch_indices.to(self.device))
            values_target = self.get_values_for_coords(target, batch_indices)

            loss = self.loss_fn(values_recon, values_target.to(self.device))
            acc_loss += loss
            count += 1

            if compute_reconstruction:
                values_recon = values_recon.cpu()
                if batch_indices.shape[1] == 2:
                    reconstructed_object[batch_indices[:, 0], batch_indices[:, 1]] = values_recon
                elif batch_indices.shape[1] == 3:
                    reconstructed_object[batch_indices[:, 0], batch_indices[:, 1], batch_indices[:, 2]] = values_recon
                else:
                    raise ValueError("Invalid number of dimensions in batch_indices")


        psnr_val = -10. * torch.log(acc_loss.cpu() / num_batches) / torch.log(torch.Tensor([10.]))

        return psnr_val, reconstructed_object


    def get_reconstructed_values(self, x):
        """
        Gets the reconstructed values from the tensor network for given coordinates.

        Parameters:
        - x: the coordinates to get the reconstructed values for.

        Returns:
        The reconstructed values and the regularization term.
        """
        if self.use_TTNF_sampling:
            input_ = get_core_tensors(self.tn)
            if self.dimensions == 2:
                coords = coord_tensor_to_coord_qtt2d(x, len(input_), chunk=True)
            elif self.dimensions == 3:
                coords = coord_tensor_to_coord_qtt3d(x, len(input_), chunk=True)
            elif self.dimensions > 3 or self.dimensions < 2: 
                raise ValueError("Only 2D and 3D supported")

            values_recon = sample_intcoord_tt_v2(input_, coords, last_core_is_payload=False, checks=True)        
            #values_recon = sample_intcoord_tt_v2(input_, coords, last_core_is_payload=False, reverse = True, checks=True)        
        else: 
            image = self.get_image_reconstruction()
            values_recon = self.get_values_for_coords(image, x)
        
        if self.activation != "None":
            values_recon = self.apply_activation(values_recon)
        
        reg_term = self.compute_regularization_term(self.tn)
        
        return values_recon, reg_term
    
    def compute_regularization_term(self, tn=None):
        """
        Computes the regularization term for the tensor network.

        Parameters:
        - tn: the tensor network.

        Returns:
        The regularization term.
        """
        if self.regularization_type == "TV":
            reg_term = self.compute_total_variation_loss(self.tn)
        elif self.regularization_type == "L1":
            reg_term = self.l1_regularization(self.tn)
        elif self.regularization_type == "L2":
            reg_term = self.l2_regularization(self.tn)
        else: 
            reg_term = 0
        
        return reg_term

    def forward(self, x, x_norm = None):
        """
        The forward pass for the model.

        Parameters:
        - x: the input tensor.
        - x_norm: normalized x values between -1 and 1 

        Returns:
        The loss for the given input.
        """
        
        values_recon, reg_term = self.get_reconstructed_values(x)

        values_target = self.get_values_for_coords(self.downsampled_target, x)

        loss =  self.loss_fn(values_recon, values_target) 

        return loss, reg_term
            