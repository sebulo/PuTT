import torch
import torch.nn as nn
import numpy as np
from utils import  downsample_with_avg_pooling, downsample_with_wavelet, downsample_with_lanczos
from save_and_plot_utils import plot_image
from PIL import Image

class BaseTNModel(torch.nn.Module):
    def __init__(self, target, init_reso, max_rank=256, dtype='float32', loss_fn_str="L2", use_TTNF_sampling=False, payload=0, payload_position="grayscale", canonization="first", activation="None", compression_alg="compress_all", regularization_type="TV", dimensions=2, noisy_target=None, device = 'cpu', masked_avg_pooling = False, sigma_init=0):
        super().__init__()

        if device == 'gpu':
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.tn = None
        self.target = target
        self.noisy_target = noisy_target
        self.init_reso = init_reso
        self.current_reso = init_reso
        self.payload = payload
        self.payload_position = payload_position
        self.dimensions = dimensions
        self.max_rank = max_rank
        self.grayscale = (payload_position == "grayscale")
        self.regularization_type = regularization_type
        self.masked_avg_pooling = masked_avg_pooling
        self.sigma_init = sigma_init
        
        self.grid_size = [init_reso for _ in range(self.dimensions)]
        if self.payload_position != 'grayscale' and self.payload != 0:
            self.grid_size.append(self.payload)
        
        print(f'payload: {self.payload}, grid_size: {self.grid_size}, device: {self.device}')

        downscale_factor = int(self.target.shape[1] / self.current_reso)
        
        self.downsampled_target = self.downsample_target(factor=downscale_factor, grayscale=self.grayscale, dim=self.dimensions, device=self.device, masked_avg_pooling = self.masked_avg_pooling)
        self.downsampled_target_non_noise = self.downsample_target(factor=downscale_factor, grayscale=self.grayscale, dim=self.dimensions, device=self.device, use_non_noisy_target=True)

        
        self.dtype = torch.float32 if dtype == 'float32' else torch.float64
        torch.set_default_dtype(self.dtype)
        
        self.num_trainable_params = 0
        self.flops_per_iter = []
        self.largest_intermediate_per_iter = []
        self.compression_factor = None
        self.loss_fn = self.get_loss_fn(loss_fn_str)


    def get_optparam_groups(self, lr_init = 0.01):
        pass

    def downsample_target(self, factor = 2, grayscale = 0, dim = 2, device = None, downsample_method = "avg_pooling", use_non_noisy_target = False, masked_avg_pooling = False):

        if device == None:
            device = self.device
        # downsample target with average pooling - we want downsampled target to be half the size of the input
        if self.noisy_target is not None and not use_non_noisy_target:
            target_tmp = self.noisy_target
            print("Using noisy target")
        else:
            target_tmp = self.target

        if factor == 1:
            res = target_tmp
            if not grayscale:
                if dim == 2:
                    res.permute(2, 0, 1)
                elif dim == 3:
                    res.permute(3, 0, 1, 2)

        # downsample_method = 'wavelet'
        if downsample_method == "avg_pooling":
            downsampled_target = downsample_with_avg_pooling(target_tmp, factor, dim, grayscale, device, masked=masked_avg_pooling)
        elif downsample_method == "wavelet":
            downsampled_target = downsample_with_wavelet(target_tmp, factor, dim)
        elif downsample_method == "lanczos":
            downsampled_target = downsample_with_lanczos(target_tmp, factor, dim)
        else:
            raise ValueError("Invalid value for downsample_method. Expected 'avg_pooling' or 'wavelet', but got {}".format(downsample_method))


        return downsampled_target.to(device)
    

    def get_loss_fn(self, loss_fn_str):
        fn_loss = {
            'L1': torch.nn.L1Loss(),
            'L2': torch.nn.MSELoss(),
            'Huber': torch.nn.HuberLoss(),
        }[loss_fn_str]
        return fn_loss
    
    def regularization(self):
        regularization_actions = {
            "None": lambda: 0,
            "TV": lambda: self.reg_term,
            "L1": self.l1_regularization,
            "L2": self.l2_regularization
        }

        try:
            return regularization_actions[self.regularization_type]()
        except KeyError:
            raise ValueError(f"Invalid value for regularization_type. Expected 'None', 'L1', 'L2', or 'TV', but got {self.regularization_type}")


    def get_values_for_coords(self, tensor, coords):
        if len(coords[0]) < 2 or len(coords[0]) > 3:
            raise ValueError("Invalid number of dimensions for coords. Expected 2 or 3, but got {}".format(len(coords[0])))
        if self.grayscale:
            if len(coords[0]) == 2:
                values = tensor[coords[:, 0], coords[:, 1]]
            elif len(coords[0]) == 3:
                values = tensor[coords[:, 0], coords[:, 1], coords[:, 2]]
        else:
            if len(coords[0]) == 2:
                values = tensor[coords[:, 0], coords[:, 1], :]
            elif len(coords[0]) == 3:
                values = tensor[coords[:, 0], coords[:, 1], coords[:, 2], :]

        return values


    def get_image_plot(self, figsize=(10,5), show_img=False):
        with torch.no_grad():
          data = self.get_image_reconstruction()
          if show_img and self.dimensions == 2:
                plot_image(data, "reconstruction", figsize)
          if show_img and self.dimensions == 3:
                # pick a slice
                plot_image(data[data.shape[2]//2, :,:], "reconstruction", figsize)
            
        return data

    def upsample(self, iteration):
        # Implementation
        pass

    def reconstruct(self):
        # Implementation
        pass

    def get_image_reconstruction(self):
        # Implementation
        pass

    def set_compression_variables(self):
        # Implementation
        pass

    

    def compute_total_variation_loss(self, img, weight=1.0):
        # Implementation
        pass

    def l1_regularization(self):
        # Implementation
        pass

    def l2_regularization(self):
        # Implementation
        pass

    def apply_activation(self, x):
        # Implementation
        pass

    def forward(self, x, x_norm = None):
        # Implementation
        pass




