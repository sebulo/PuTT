# Based on TensoRF code from https://github.com/apchenstu/TensoRF
# Modifications and/or extensions have been made for specific purposes in this project.


from .tensorBase import *
import math

from opt_einsum import contract, contract_path

from functools import partial

from models.quimb.tn_utils import sample_generic_3d, sample_intcoord_tensor3d, coords_to_v2_tensor_ring, sample_intcoord_TR_3d_v2

class TensorRing(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        print("kargs", kargs)
        super(TensorRing, self).__init__(aabb, gridSize, device, **kargs)
        
        self.checks = False
        self.vecMode = [0,1,2] 
        #self.vecMode = [2,1,0] 
        self.reverse_coords = True if self.vecMode[0] > self.vecMode[1] else False
        print("self.reverse_coords", self.reverse_coords)
        
        
    def optimize_contraction_expression_density(self):
        
        app_components = self.app_components
        density_components = self.density_components
        
        # self.einsum_str_app = 'ixj,jyk,kzi,ici->xyzc' # THIS WORKS suprisingly well better than 'ixj,jyk,kzp,pci->xyzc'
        self.einsum_str_app = 'ixj,jyk,kzp,pci->xyzc'
        self.einsum_str_density = 'ixj,jyk,kzi->xyz'
        
        if not self.is_tensor_ring: # is regular tensor train
            self.einsum_str_app = 'cxj,jyk,kzc->xyzc'
            self.einsum_str_density = 'ixj,jyk,kzi->xyz'
            
        self.einsum_path_app = self.optimize_contraction_expression(app_components, self.einsum_str_app)
        self.einsum_path_density = self.optimize_contraction_expression(density_components, self.einsum_str_density)
        
    def optimize_contraction_expression(self, components, einsum_str):
        path_info = contract_path(einsum_str, *components, optimize='optimal')
        self.path_info = path_info
        return path_info[0]
        
    def init_svd_volume(self, res, device):
        app_components = self.init_one_svd(self.app_n_comp, self.gridSize, self.init_scale, self.app_dim, device)
        density_components = self.init_one_svd(self.density_n_comp, self.gridSize, self.init_scale, 0, device, is_density = True)
        
        # make parameter list
        self.app_components = torch.nn.ParameterList(app_components)
        self.density_components = torch.nn.ParameterList(density_components)
        
        if not self.use_TTNF_sampling:
            self.optimize_contraction_expression_density()

        self.print_tensor_ring_size()
        
    def print_tensor_ring_size(self):
        for i in range(len(self.app_components)):
            print(f'app component {i}: {self.app_components[i].shape}')
        for i in range(len(self.density_components)):
            print(f'density component {i}: {self.density_components[i].shape}')
            
        
    def init_one_svd(self, n_component, gridSize, scale, payload, device, is_density = False):
        if not self.is_tensor_ring:
            return self.init_one_tensor_train(n_component, gridSize, scale, payload, device, is_density)
        
        components = []
        is_first = True
        for i in range(len(self.vecMode)):
            if is_first:
                component = torch.nn.Parameter(scale * torch.randn((n_component[-1], gridSize[self.vecMode[i]], n_component[i])).to(device))
                is_first = False
            elif i == len(self.vecMode) - 1:
                component = torch.nn.Parameter(scale * torch.randn((n_component[i-1], gridSize[self.vecMode[i]], n_component[-1])).to(device))
            else:
                component = torch.nn.Parameter(scale * torch.randn((n_component[i-1], gridSize[self.vecMode[i]], n_component[i])).to(device))
            
            components.append(component)
            
        if is_density:
            return components
        
        channel_component = torch.nn.Parameter(scale * torch.randn((n_component[-1], payload, n_component[-1])).to(device))
        
        return components + [channel_component]

    def init_one_tensor_train(self, n_component, gridSize, scale, payload, device, is_density = False):
        last_dim = 1 if is_density else payload
        components = []
        is_first = True
        for i in range(len(self.vecMode)):
            if is_first:
                component = torch.nn.Parameter(scale * torch.randn((last_dim, gridSize[self.vecMode[i]], n_component[i])).to(device))
                is_first = False
            elif i == len(self.vecMode) - 1:
                component = torch.nn.Parameter(scale * torch.randn((n_component[i-1], gridSize[self.vecMode[i]], last_dim)).to(device))
            else:
                component = torch.nn.Parameter(scale * torch.randn((n_component[i-1], gridSize[self.vecMode[i]], n_component[i])).to(device))
            
            components.append(component)
            
        return components

        

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        
        grad_vars = [{'params': self.density_components, 'lr': lr_init_spatialxyz},
                     {'params': self.app_components, 'lr': lr_init_spatialxyz}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
            
        return grad_vars
    
    def density_L1(self):
        loss = 0
        for comp in self.density_components:
            loss += torch.mean(torch.abs(comp))
        return loss
        
    def TV_loss_density(self, reg):
        loss = 0
        tv_components = self.density_components
        for comp in tv_components:
            # each is tensor of shape (r1, n, r2) - do difference along the n dimension
            loss += torch.sum(torch.abs(comp[:, 1:, :] - comp[:, :-1, :]))
        return reg * loss

    def TV_loss_app(self, reg):
        loss = 0
        tv_components = self.app_components
        for comp in tv_components:
            # each is tensor of shape (r1, n, r2) - do difference along the n dimension
            loss += torch.sum(torch.abs(comp[:, 1:, :] - comp[:, :-1, :]))
        return reg * loss
    
    def contract_tensor_network_denisty(self):
        return contract(self.einsum_str_density, *self.density_components, optimize=self.einsum_path_density)
        #return torch.einsum(self.einsum_str_density, *self.density_components)
    
    def contract_tensor_network_app(self):
        return contract(self.einsum_str_app, *self.app_components, optimize=self.einsum_path_app)
        #return torch.einsum(self.einsum_str_app, *self.app_components)
    
    def compute_appfeature(self, coords_xyz):
        if self.use_TTNF_sampling:
            # input_ = self.app_components  but this is a parameter list
            input_ = [d.data for d in self.app_components]
            print("input_ len", len(input_))
            print("input_ 0", input_[0].shape)
            print("input_ 1", input_[1].shape)
            print("input_ 2", input_[2].shape)
            last_core_is_payload = True
            fn_sample_intcoord = partial(sample_intcoord_TR_3d_v2, last_core_is_payload=last_core_is_payload)
            sample_redundancy_handling = True
        else:
            input_ = self.contract_tensor_network_app()
            #fn_sample_intcoord = sample_intcoord_tensor3d
            fn_sample_intcoord = partial(sample_intcoord_tensor3d, reverse=self.reverse_coords)
            sample_redundancy_handling = False

        out =  sample_generic_3d(
            input_,
            coords_xyz,
            fn_sample_intcoord=fn_sample_intcoord,
            sample_redundancy_handling = sample_redundancy_handling,
            outliers_handling="zeros",
            has_channel_comp = True if len(self.app_components) > 3 else False,
            checks=self.checks
        )
        
        return out 
    
    def compute_densityfeature(self, coords_xyz):
        if self.use_TTNF_sampling:
            # input_ = self.density_components
            input_ = [d.data for d in self.density_components]
            print("input_ len", len(input_))
            print("input_ 0", input_[0].shape)
            print("input_ 1", input_[1].shape)
            print("input_ 2", input_[2].shape)
            #last_core_is_payload = True if self.use_channel_comp_density else False
            last_core_is_payload = False
            fn_sample_intcoord = partial(sample_intcoord_TR_3d_v2, last_core_is_payload=last_core_is_payload)
            sample_redundancy_handling = True
        else:
            input_ = self.contract_tensor_network_denisty().unsqueeze(-1)
            #fn_sample_intcoord = sample_intcoord_tensor3d
            fn_sample_intcoord = partial(sample_intcoord_tensor3d, reverse=self.reverse_coords)
            sample_redundancy_handling = False

        out =  sample_generic_3d(
            input_,
            coords_xyz,
            fn_sample_intcoord=fn_sample_intcoord,
            sample_redundancy_handling = sample_redundancy_handling,
            outliers_handling="zeros",
            has_channel_comp = False,
            checks=self.checks
        )
        
        return out.squeeze()
        
    
    def get_compression_values(self):
        self.num_uncompressed_params = torch.prod(self.gridSize) * (self.app_dim + 1)
        num_compressed_params = 0
        
        for tensor in self.density_components:
            num_compressed_params += math.prod(tensor.shape)
            
        for tensor in self.app_components:
            num_compressed_params += math.prod(tensor.shape)
            
        self.num_compressed_params = num_compressed_params
        self.compression_factor = self.num_uncompressed_params / self.num_compressed_params
        
        # get size in GB
        float_type = self.density_components[0].dtype
        if float_type == torch.float64:
            bytes_per_float = 8
        if float_type == torch.float32:
            bytes_per_float = 4
        elif float_type == torch.float16:
            bytes_per_float = 2
        else:
            print("ERROR: unknown float type")
            bytes_per_float = 4
        print(f'bytes per float: {bytes_per_float}')
        print(f'compressed size: {num_compressed_params * bytes_per_float / 1024 ** 2} MB')
        
        self.sz_uncompressed_gb = self.num_uncompressed_params * bytes_per_float / 1024**3
        self.sz_compressed_gb = num_compressed_params * bytes_per_float / 1024 ** 3
        
        return {
            'num_uncompressed_params': self.num_uncompressed_params,
            'num_compressed_params': self.num_compressed_params,
            'sz_uncompressed_gb': self.sz_uncompressed_gb,
            'sz_compressed_gb': self.sz_compressed_gb,
            'compression_factor': self.compression_factor
        }
    
    @torch.no_grad()
    def upsampling_TT(self, components, res_target = [32, 32, 32], squeeze_last = True):
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
            
            component_interpolated = component_interpolated.squeeze(0)
            
            # Move the upsampled component to the specified device
            return torch.nn.Parameter(component_interpolated).to(self.device)
        
        for i in range(len(self.vecMode)): # does not include channel component
            components[i] = upsample_component_trilinear(components[i], res_target, self.vecMode[i])
        
        if squeeze_last:
            for i in range(len(self.vecMode)):
                components[i] = components[i].squeeze(-1)


        return components
    
   
    def to_device(self, tensor_list, device):
        """
        Move a list of tensors or a ParameterList to a specified device.
        
        Args:
        tensor_list (list of torch.Tensor or torch.nn.ParameterList): The list of tensors or ParameterList to move.
        device (torch.device or str): The target device.
        
        Returns:
        list of torch.Tensor or torch.nn.ParameterList: The list of tensors or ParameterList moved to the target device.
        """
        if isinstance(tensor_list, torch.nn.ParameterList):
            # If the input is a ParameterList, handle each parameter individually
            return torch.nn.ParameterList([tensor.to(device) for tensor in tensor_list])
        else:
            # Handle as a regular list
            return [tensor.to(device) for tensor in tensor_list]

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        """
        Upsample volume grid components to a target resolution.
        
        Args:
        res_target (int): The target resolution for upsampling.
        """
        # Move components to CPU
        self.app_components = self.to_device(self.app_components, 'cpu')
        self.density_components = self.to_device(self.density_components, 'cpu')

        # Perform upsampling on CPU
        density_components = self.upsampling_TT(self.density_components, res_target) 
        app_components = self.upsampling_TT(self.app_components, res_target)
        
        self.app_components = torch.nn.ParameterList(app_components)
        self.density_components = torch.nn.ParameterList(density_components)    
        
        # Move components back to the original device
        self.app_components = self.to_device(self.app_components, self.device)
        self.density_components = self.to_device(self.density_components, self.device)
        
        if not self.use_TTNF_sampling:
            self.optimize_contraction_expression_density()
        
        # Additional method calls
        self.print_tensor_ring_size()
        self.update_stepSize(res_target)
        print(f'Upsampling to {res_target}')

    
    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> Updating aabb ...")
        print("old aabb", self.aabb)
        self.aabb = new_aabb
        print("new aabb", self.aabb)
        print('self.should_shrink', self.should_shrink)
        if self.should_shrink:
            print("====> shrinking ...")
            
            xyz_min, xyz_max = new_aabb
            t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
            t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
            b_r = torch.stack([b_r, self.gridSize]).amin(0)
            
            print(f't_l: {t_l}, b_r: {b_r}')
            
            app_components = []
            density_components = []
            print("vecMode", self.vecMode)
            for i in range(len(self.vecMode)):
                vec_mode = self.vecMode[i]
                app_component = torch.nn.Parameter(self.app_components[i].data[..., t_l[vec_mode]:b_r[vec_mode], :])
                app_components.append(app_component.to(self.device))
                density_component = torch.nn.Parameter(self.density_components[i].data[..., t_l[vec_mode]:b_r[vec_mode], :])
                density_components.append(density_component.to(self.device))
            
            # add app component for channel
            if self.is_tensor_ring:
                app_components.append(torch.nn.Parameter(self.app_components[-1].data).to(self.device))
            self.app_components = torch.nn.ParameterList(app_components)
            self.density_components = torch.nn.ParameterList(density_components)
            
            print("self.alphaMask.gridSize ",self.alphaMask.gridSize )
            print("self.gridSize ",self.gridSize)
            if not torch.all(self.alphaMask.gridSize == self.gridSize):
                t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
                correct_aabb = torch.zeros_like(new_aabb)
                correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
                correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
                print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
                new_aabb = correct_aabb
                
            if not self.use_TTNF_sampling:
                self.optimize_contraction_expression_density()
                
            self.gridSize = b_r - t_l
            self.print_tensor_ring_size()
            self.update_stepSize(self.gridSize)
            print("====> shrinking done")
        