# Based on TT-NF code from https://github.com/toshas/ttnf
# Modifications and/or extensions have been made for specific purposes in this project.

import torch
import os
import numpy as np
import quimb.tensor as qtn
import quimb as quimb
import time
# from same directory import tn_utils
from .tn_utils import get_qtt_TTNF, get_core_tensors, prolongate_qtt, sample_generic_3d, sample_intcoord_qtt3d_v2,qtt_to_tensor, sample_intcoord_tensor3d, increase_ranks
from skimage import data
import matplotlib.pyplot as plt



class QTT3dQuimb(torch.nn.Module):

    def __init__(self, init_reso, max_rank_tt = 256,  dtype = 'float32', use_TTNF_sampling = True, update_model_specs = True, payload_dim = 0,
                 payload_position = 'first_core', outliers_handling = "zeros", checks =False,
                compression_alg = "compress_all", canonization = "None"):
        """
        Initializes the QTT3dQuimb object for 3D Quimb Tensor Train representation.

        Parameters:
        - init_reso: Initial resolution of the tensor.
        - max_rank_tt: Maximum rank for tensor train decompositions.
        - dtype: Data type for computations (default 'float32').
        - use_TTNF_sampling: Whether to use TTNF sampling.
        - update_model_specs: Boolean indicating if model specifications should be updated.
        - payload_dim: Additional payload dimensions.
        - payload_position: The position of the payload in the tensor network - either 'first_core' or 'last_core'.
        - outliers_handling: Method for handling outliers, default is "zeros".
        - checks: Boolean to perform checks.
        - compression_alg: Algorithm for tensor compression.
        - canonization: Method for canonization in tensor network.
        """
        super().__init__()
        self.current_reso = init_reso
        self.max_rank_tt = max_rank_tt
        self.inds = 'k'
        self.use_TTNF_sampling = use_TTNF_sampling
        self.payload_dim = payload_dim
        self.payload_position = payload_position
        self.compression_alg = compression_alg
        self.canonization = canonization

        self.dim_grid_log2 = int(np.log2(init_reso))
        self.tn = None
        self.shape_source = None
        self.shape_target = None
        self.shape_factors = None
        self.factor_target_to_source = None

        # from TTNF
        self.outliers_handling= outliers_handling
        self.checks= checks

        self.update_model_specs = True
        self.num_trainable_params = 0
        self.flops_per_iter = []
        self.largest_intermediate_per_iter = []
      
        self.init_tn()# extract the raw arrays and a skeleton of the TN
        

        # set default dtype based on dtype args
        self.dtype = torch.float64 if dtype == 'float64' else torch.float32
        torch.set_default_dtype(self.dtype)

        self.compression_factor = None
        self.set_compression_variables()

        # get equation for contraction
        if not self.use_TTNF_sampling:
            self.contraction_expression, self.path_info, self.symbol_info = self.get_contraction_expression()
        else:
            self.contraction_expression = None
            self.path_info = None
            self.symbol_info = None

    @torch.no_grad()
    def get_contraction_expression(self, only_expression = False):
        """
        Computes the contraction expression for the tensor network.

        Parameters:
        - only_expression: Boolean indicating whether to return only the contraction expression.

        Returns:
        A tuple containing the contraction expression, path information, and symbol information.
        """
        output_inds = self.tn.outer_inds()
        output_inds = [ind for ind in output_inds if ind != 'payload'] # remove string "payload" from output inds 
        output_inds = ['payload'] + output_inds # add to front of output inds
        backend = 'torch'
        optimze = 'auto-hq'
        if only_expression:
            contraction_expression = self.tn.contract(output_inds=output_inds, get='expression', backend=backend, optimize=optimze)
            return contraction_expression, None, None
        
        path_info =self.tn.contract( output_inds=output_inds, get='path-info', backend=backend, optimize=optimze)
        self.update_contraction_costs(path_info)
        symbol_info =self.tn.contract( output_inds=output_inds, get='symbol-map', backend = backend, optimize=optimze)
        contraction_expression = self.tn.contract(output_inds=output_inds, get='expression', backend=backend, optimize=optimze)

        return contraction_expression, path_info, symbol_info

    def init_tn(self):
        """
        Initializes the tensor network (TN) for the model.
        """
        tn, self.shape_source, self.shape_target, self.shape_factors, _, self.factor_target_to_source = get_qtt_TTNF(self.current_reso, self.max_rank_tt, dim=3, payload_dim=self.payload_dim, payload_position=self.payload_position, compression_alg=self.compression_alg, canonization=self.canonization)
        print("QTT: ", self.tn)
        self.torch_params, self.skeleton = self.get_torch_params(tn)
        self.tn = self.reconstruct()

    @torch.no_grad()
    def get_torch_params(self, tn):
        """
        Retrieves the parameters of the tensor network as a PyTorch ParameterDict.

        Parameters:
        - tn: The tensor network from which to extract parameters.

        Returns:
        A tuple containing the PyTorch ParameterDict and the skeleton of the tensor network.
        """
        params, self.skeleton = qtn.pack(tn)
        self.torch_params = torch.nn.ParameterDict({
            # torch requires strings as keys
            str(i): torch.nn.Parameter(initial)
            for i, initial in params.items()
        })
        return self.torch_params, self.skeleton
    
    
    @torch.no_grad()
    def upsample_ranks(self, new_max_rank):
        
        tn = increase_ranks(self.tn, new_max_rank, payload_position = self.payload_position)
        
        self.get_torch_params(tn)
        self.tn = self.reconstruct()
        
        if self.update_model_specs:
            self.set_compression_variables()
            self.update_contraction_costs()
            self.num_trainable_params = sum(p.numel() for p in self.torch_params.values())
            print("Number trainable parameters", self.num_trainable_params)
        

        if not self.use_TTNF_sampling:
            self.contraction_expression, _, _= self.get_contraction_expression(only_expression=True)
        

    @torch.no_grad()
    def upsample(self, iteration):
        """
        Upsamples the tensor network.

        Parameters:
        - iteration: The current iteration number, used in the upsampling process to determine the indices of the tensor network.
        """
        
        self.current_reso = self.current_reso * 2
        self.dim_grid_log2 = self.dim_grid_log2 + 1

        tn, self.shape_source, self.shape_factors, self.factor_target_to_source = prolongate_qtt(self.tn, dim=3, ranks_tt= self.max_rank_tt, payload_position = self.payload_position, compression_alg = self.compression_alg, canonization = self.canonization)
        self.inds = 'b' + str(iteration) # used for computing output inds TODO: CHECK if this works again
    
        self.get_torch_params(tn)
        self.tn = self.reconstruct()
        
        if self.update_model_specs:
            self.set_compression_variables()
            self.update_contraction_costs()
            self.num_trainable_params = sum(p.numel() for p in self.torch_params.values())
            print("Number trainable parameters", self.num_trainable_params)
        

        if not self.use_TTNF_sampling:
            self.contraction_expression, _, _= self.get_contraction_expression(only_expression=True)


    def reconstruct(self):
        """
        Reconstructs the tensor network as a quimb tensor network.

        Returns:
        The reconstructed tensor network.
        """
        # convert back to original int key format
        params = {int(i): p for i, p in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        return psi

    def set_compression_variables(self):
        """
        Sets the compression variables for the tensor network.
        """
        self.dtype_sz_bytes = {
            torch.float16: 2,
            torch.float32: 4,
            torch.float64: 8,
        }[self.dtype]
        self.num_uncompressed_params =  np.prod(self.shape_source) * self.payload_dim
        self.num_compressed_params = sum(p.numel() for p in self.torch_params.values())
        self.sz_uncompressed_gb = self.num_uncompressed_params * self.dtype_sz_bytes / (1024 ** 3)
        self.sz_compressed_gb = self.num_compressed_params * self.dtype_sz_bytes / (1024 ** 3)
        self.compression_factor = self.num_uncompressed_params / self.num_compressed_params 

    @torch.no_grad()
    def update_contraction_costs(self, contraction_info = None):
        """
        Updates the cost of contraction variables for the tensor network - usually called after upsampling.

        Parameters:
        - contraction_info: The contraction information to use (if None, computes it from the current network).
        """
        if contraction_info is None:
            contraction_info = self.tn.contraction_info()
        self.flops = contraction_info.opt_cost
        self.largest_intermediate = contraction_info.largest_intermediate
        self.flops_per_iter = self.flops_per_iter + [ self.flops]
        self.largest_intermediate_per_iter = self.largest_intermediate_per_iter + [ self.largest_intermediate]
        print("FLOPS", self.flops)
        print("Largest intermediate", self.largest_intermediate)
    
    def get_tensor_representation(self):
        """
        Gets the dense tensor representation of the tensor network.

        Returns:
        The tensor representation of the tensor network.
        """
        return qtt_to_tensor(self.tn, self.shape_source, self.shape_factors, self.factor_target_to_source,
                                 inds=self.inds, payload_position=self.payload_position, payload = self.payload_dim,
                                    expression=self.contraction_expression)
    
    def compute_total_variation_loss(self):
        """
        Computes the total variation loss for the tensor network.

        Parameters:
        - tn: the tensor network with a number of cores len(tn.tensors)

        Returns:
        The total variation loss.
        """
        total_variation_loss = 0
        for tensor in self.tn.tensors:
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
    
    def save_tn(self, base_path):
        """
        Saves the current tensor network's parameters and skeleton to disk.

        Parameters:
        - base_path: Base path for saving the files.

        Returns:
        Paths for the saved parameter and skeleton files.
        """
        params_path = base_path + "_params.pth"  # Parameters file
        skeleton_path = base_path + "_skeleton"  # Skeleton file

        torch.save(self.torch_params, params_path)
        quimb.save_to_disk(self.skeleton, skeleton_path)
        
        return params_path, skeleton_path

    
    def load_tn(self, params, skeleton):
        """
        Loads the tensor network parameters and skeleton.
        Also updates the compression variables and contraction costs.

        Parameters:
        - params: Parameters of the tensor network.
        - skeleton: Skeleton of the tensor network.

        Returns:
        The loaded tensor network.
        """
        unpacked_params = {int(i): p for i, p in params.items()} #
        self.tn = qtn.unpack(unpacked_params, skeleton)
        self.set_compression_variables()
        self.update_contraction_costs()
        self.num_trainable_params = sum(p.numel() for p in self.torch_params.values())
        self.contraction_expression, self.path_info, self.symbol_info = self.get_contraction_expression()
        
        # set as parameters
        self.get_torch_params(self.tn)
        return self.tn


    def forward(self, coords_xyz):
        """
        Forward pass for the model. Samples values from the tensor network based on provided coordinates.
        if use_TTNF_sampling is True, samples from the TTNF, otherwise samples from the full tensor obtained by contracting the QTT.

        Parameters:
        - coords_xyz: Coordinates to sample from the tensor network.

        Returns:
        Sampled values from the tensor network.
        """

        if self.use_TTNF_sampling:
            input = get_core_tensors(self.tn)
            fn_sample_intcoord = sample_intcoord_qtt3d_v2
            sample_redundancy_handling = True
        else: # sample from the full tensor by first contracting the TT
            input = qtt_to_tensor(self.tn, self.shape_source, self.shape_factors, self.factor_target_to_source, 
                                  inds=self.inds, payload_position=self.payload_position, payload = self.payload_dim,
                                  expression=self.contraction_expression)
            fn_sample_intcoord = sample_intcoord_tensor3d
            sample_redundancy_handling = False

        out =  sample_generic_3d(
            input,
            coords_xyz,
            fn_sample_intcoord=fn_sample_intcoord,
            sample_redundancy_handling = sample_redundancy_handling,
            outliers_handling=self.outliers_handling,
            checks=self.checks
        )
        return out