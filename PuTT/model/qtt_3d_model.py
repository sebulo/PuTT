import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from ..core.tt_core import *

from PuTT.tn_utils import *


class QTT3dQuimb(torch.nn.Module):

    def __init__(self, init_reso, max_rank = 256,  dtype = 'float64', loss_fn_str = "L2", use_TTNF_sampling = True, update_model_specs = False, payload = 0, payload_position = 'first_core'):
        """
        """
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.current_reso = init_reso
        self.max_rank = max_rank
        self.inds = 'k'
        self.use_TTNF_sampling = use_TTNF_sampling
        self.payload = payload
        self.payload_position = payload_position
        self.grayscale = True if self.payload_position == 'grayscale' else False

        self.dim_grid_log2 = int(np.log2(init_reso))
        self.tn = None
        self.shape_source = None
        self.shape_target = None
        self.shape_factors = None
        self.factor_target_to_source = None

        self.update_model_specs = update_model_specs
        self.num_trainable_params = 0
        self.flops_per_iter = []
        self.largest_intermediate_per_iter = []
      
        self.init_tn()

        # extract the raw arrays and a skeleton of the TN
        self.torch_params, self.skeleton = self.get_torch_params(self.tn)

        # set default dtype based on dtype args
        self.dtype = torch.float64 if dtype == 'float64' else torch.float32
        torch.set_default_dtype(self.dtype)

        self.compression_factor = None
        if update_model_specs:
            self.set_compression_variables()
            self.update_contraction_costs()

    def init_tn(self):
        self.tn, self.shape_source, self.shape_target, self.shape_factors, _, self.factor_target_to_source = get_qtt_QTTNF(self.current_reso, self.max_rank, dim=3, payload=self.payload, payload_position=self.payload_position)

    def get_torch_params(self, tn):
        params, self.skeleton = qtn.pack(tn)
        self.torch_params = torch.nn.ParameterDict({
            # torch requires strings as keys
            str(i): torch.nn.Parameter(initial)
            for i, initial in params.items()
        })
        trainable_params = sum(p.numel() for p in self.torch_params.values())
        print("Number of parameters", trainable_params)
        return self.torch_params, self.skeleton


    def upsample(self, iteration):
        self.current_reso = self.current_reso * 2
        self.dim_grid_log2 = self.dim_grid_log2 + 1
        if iteration == 0:
            self.tn, self.shape_source, self.shape_factors, self.factor_target_to_source = prolongate_qtt(self.tn, dim=2, ranks_tt= self.max_rank, lower_ind='b0', payload_position = self.payload_position, device = self.device)
            self.inds = 'b0'
        else: 
            self.tn, self.shape_source, self.shape_factors, self.factor_target_to_source = prolongate_qtt(self.tn, dim=2, ranks_tt= self.max_rank , upper_ind='b' + str(iteration-1), lower_ind='b' + str(iteration), payload_position = self.payload_position, device = self.device)
            self.inds = 'b' + str(iteration)
        self.get_torch_params(self.tn)
        if self.update_model_specs:
            self.set_compression_variables()
            self.update_contraction_costs()
            self.num_trainable_params = sum(p.numel() for p in self.torch_params.values())
            print("Number trainable parameters", self.num_trainable_params)


    def reconstruct(self):
        # convert back to original int key format
        params = {int(i): p for i, p in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        return psi

    def get_3d_reconstruction(self):
        psi = self.reconstruct()
        data = qtt_to_tensor(psi, self.shape_source, self.shape_factors, self.factor_target_to_source, inds=self.inds, payload_position=self.payload_position, grayscale= self.grayscale, payload=self.payload) # TOO rename to qtt_to_tensor
        return data
          
    def set_compression_variables(self):
        self.dtype_sz_bytes = {
            torch.float16: 2,
            torch.float32: 4,
            torch.float64: 8,
        }[self.dtype]
        num_channels = 1 if self.payload_position == 'grayscale' else 3
        self.num_uncompressed_params = self.current_reso**num_channels
        self.num_compressed_params = sum(p.numel() for p in self.torch_params.values())
        self.sz_uncompressed_gb = self.num_uncompressed_params * self.dtype_sz_bytes / (1024 ** 3)
        self.sz_compressed_gb = self.num_compressed_params * self.dtype_sz_bytes / (1024 ** 3)
        self.compression_factor = self.num_uncompressed_params / self.num_compressed_params

    
    def update_contraction_costs(self):
        contraction_info = self.tn.contraction_info()
        self.flops = contraction_info.opt_cost
        self.largest_intermediate = contraction_info.largest_intermediate
        self.flops_per_iter = self.flops_per_iter + [ self.flops]
        self.largest_intermediate_per_iter = self.largest_intermediate_per_iter + [ self.largest_intermediate]
        print("FLOPS", self.flops)
        print("Largest intermediate", self.largest_intermediate)
    

    def get_voxel_values_for_coords(self, data, coords_xyz):
        if self.grayscale:
            values = data[coords_xyz[:,0], coords_xyz[:,1], coords_xyz[:,2]]
        else: 
            values = data[coords_xyz[:,0], coords_xyz[:,1], coords_xyz[:,2], :]
        return values

    def forward(self, coords_xyz):
        if self.use_TTNF_sampling:
            psi = self.reconstruct()
            input = get_core_tensors(psi)
            coords = coord_tensor_to_coord_qtt2d(coords_xyz, len(input), chunk=True)
            # THIS FAILS
            # coords = coords.to(self.device)
            values_recon = sample_intcoord_tt_v2(input, coords, last_core_is_payload=False)        
        else: 
            data = self.get_3d_reconstruction()
            values_recon = self.get_voxel_values_for_coords(data, coords_xyz)

        return values_recon
    