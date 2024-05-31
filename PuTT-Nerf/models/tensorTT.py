from .quimb.qtt_3d_model import *
from .quimb.tn_utils import *
from .tensorBase import *

class TensorTT(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        """
        Initializes the TensorTT object.

        Parameters:
        aabb (array): Axis-aligned bounding box defining the volume boundaries.
        gridSize (tuple): The size of the grid.
        device (str): The device (e.g., 'cpu' or 'cuda') on which computations will be performed.
        **kargs: Additional keyword arguments.
        """
        super(TensorTT, self).__init__(aabb, gridSize, device, **kargs)
        self.num_upsamples_perfomed = 0
        print("====> init TensorTT - fused: ", self.fused)
    

    def init_svd_volume(self, res, device):
        """
        Initializes the QTT tensors.
        
        Parameters:
        res (int): side length of the volume grid.
        device (str): The device on which computations will be performed.
        """
        self.init_res = res
        # max_rank_appearance and max_rank_density and max_rank are the possibilities
        if self.fused:
            payload = self.app_dim + 1
            self.vox_fused = QTT3dQuimb(res, max_rank_tt = self.max_rank, use_TTNF_sampling= self.use_TTNF_sampling,
                                        payload_dim = payload,
                                        compression_alg = self.compression_alg, canonization = self.canonization)
        else:
            max_rank_appearance = self.max_rank_appearance if self.max_rank_appearance > 0 else self.max_rank
            max_rank_density = self.max_rank_density if self.max_rank_density > 0 else self.max_rank

            self.vox_rgb = QTT3dQuimb(res, max_rank_tt = max_rank_appearance, use_TTNF_sampling= self.use_TTNF_sampling,
                                    payload_dim = self.app_dim,
                                    compression_alg = self.compression_alg, canonization = self.canonization)  # color
            self.vox_sigma = QTT3dQuimb(res, max_rank_tt = max_rank_density, use_TTNF_sampling=self.use_TTNF_sampling,
                                    payload_dim = 1,
                                    compression_alg = self.compression_alg, canonization = self.canonization)
        
        self.take_voxel_representations_on_device(device)
        

    def take_voxel_representations_on_device(self, device):
        """
        Transfers the voxel representations to the specified device.

        Parameters:
        device (str): The device to transfer the voxel representations to.
        """
        if self.fused:
            self.vox_fused = self.vox_fused.to(device) 
        else:
            self.vox_rgb = self.vox_rgb.to(device)
            self.vox_sigma = self.vox_sigma.to(device)

    def compute_features(self, xyz_sampled):
        """
        Computes features for the given sampled coordinates.

        Parameters:
        xyz_sampled (array): Sampled coordinates with shape (N, 3).

        Returns:
        tuple: The computed RGB and sigma values with shapes (N, 3) and (N,).
        """
        if self.fused:
            res =  self.vox_fused(xyz_sampled)
            # devide into rgb and sigma
            rgb = res[:, :-1]
            sigma = res[:, -1:]
            return rgb, sigma.squeeze()
        else:
            rgb = self.vox_rgb(xyz_sampled)
            sigma = self.vox_sigma(xyz_sampled)
            return rgb, sigma.squeeze()

    
    def compute_densityfeature(self, xyz_sampled):
        """
        Computes the density feature for the given sampled coordinates.

        Parameters:
        xyz_sampled (array): Sampled coordinates with shape (N, 3).

        Returns:
        array: The computed density values with shape (N,).
        """
        # print min max of xyz_sampled
        if self.fused:
            _, sigma = self.compute_features(xyz_sampled)
            return sigma.squeeze()
        return self.vox_sigma(xyz_sampled).squeeze()
    
    def compute_appfeature(self, xyz_sampled):
        """
        Computes the appearance feature for the given sampled coordinates.

        Parameters:
        xyz_sampled (array): Sampled coordinates with shape (N, 3).

        Returns:
        array: The computed RGB values with shape (N, 3).
        """
        if self.fused:
            rgb, _ = self.compute_features(xyz_sampled)
            return rgb
        return self.vox_rgb(xyz_sampled)
    
    def density_L1(self):
        """
        Calculates the L1 norm of the density tensors.

        Returns:
        float: The computed L1 norm.
        """
        total = 0
        tensors = self.vox_sigma.tn.tensors
        for i in range(len(tensors)):
            total += torch.mean(torch.abs(tensors[i].data))
        return total


    def TV_loss(self, density_only = False):
        """
        Computes the total variation loss for the volume grid of either the RGB or the density tensors.
        """
        if density_only:
            return self.vox_sigma.compute_total_variation_loss()
        else:
            return self.vox_rgb.compute_total_variation_loss() 
    
    def get_max_ranks(self):
        """
        Retrieves the maximum ranks of the volume grid.

        Returns:
        tuple: The maximum ranks of the RGB and the density tensors.
        """
        if self.fused:
            return self.vox_fused.tn.max_bond(), self.vox_fused.tn.max_bond()
        else:
            return self.vox_rgb.tn.max_bond(), self.vox_sigma.tn.max_bond()
        
    def get_norms(self):
        """
        Retrieves the Frobenius norms of the volume grid.

        Returns:
        tuple: The Frobenius norms of the RGB and the density tensors.
        """
        if self.fused:
            # return self.vox_fused.tn.H @ self.vox_fused.tn 
            return self.vox_fused.tn.norm(), self.vox_fused.tn.norm()
        else:
            # norm_vox_rgb = self.vox_rgb.tn.H @ self.vox_rgb.tn
            # norm_vox_sigma = self.vox_sigma.tn.H @ self.vox_sigma.tn
            # return norm_vox_rgb, norm_vox_sigma
            return self.vox_rgb.tn.norm(), self.vox_sigma.tn.norm()
    
    @torch.no_grad()
    def upsample_volume_ranks(self, max_rank_appearance, max_rank_density):
        """
        Upsamples the volume grid ranks to the target rank.

        Parameters:
        max_rank (int): Target rank.
        """
        if self.fused:
            self.vox_fused.upsample_ranks(max_rank_appearance)
            self.vox_fused.max_rank_tt = max_rank_appearance
        else:
            self.vox_rgb.upsample_ranks(max_rank_appearance)
            self.vox_sigma.upsample_ranks(max_rank_density)
            self.vox_rgb.max_rank_tt = max_rank_appearance
            self.vox_sigma.max_rank_tt = max_rank_density
            
            print(" sigma", self.vox_sigma.tn)
            print(" rgb", self.vox_rgb.tn)
            
    
    @torch.no_grad()
    def upsample_volume_grid(self, reso_target):
        """
        Upsamples the volume grid to the target resolution.

        Parameters:
        reso_target (int): Target resolution.
        """
        if self.fused:
            self.vox_fused.upsample(self.num_upsamples_perfomed)
        else:
            self.vox_rgb.upsample(self.num_upsamples_perfomed)
            self.vox_sigma.upsample(self.num_upsamples_perfomed)
        self.num_upsamples_perfomed += 1


        self.update_stepSize(reso_target)
        print(f'upsamping to {reso_target}')


    def get_optparam_groups(self, lr_init_spatial = 0.003, lr_init_network = 0.001):
        """
        Groups the optimization parameters.

        Parameters:
        lr_init_spatial (float): Initial learning rate for spatial parameters.
        lr_init_network (float): Initial learning rate for network parameters.

        Returns:
        list: List of dictionaries containing parameters and their learning rates.
        """
        out = []
        if self.fused:
            out += [
                {'params': self.vox_fused.parameters(), 'lr': lr_init_spatial}
            ]
        else:
            out += [
                {'params': self.vox_rgb.parameters(), 'lr': lr_init_spatial},
                {'params': self.vox_sigma.parameters(), 'lr': lr_init_spatial}
            ]
        if isinstance(self.renderModule, torch.nn.Module):
            out += [
                {'params': self.renderModule.parameters(), 'lr': lr_init_network}
            ]
        return out

    def get_compression_values(self):
        """
        Retrieves various compression values.

        Returns:
        dict: A dictionary containing uncompressed and compressed parameters, sizes, and compression factor.
        """
        if self.fused:
            self.num_uncompressed_params = self.vox_fused.num_uncompressed_params
            self.num_compressed_params = self.vox_fused.num_compressed_params
            self.sz_uncompressed_gb = self.vox_fused.sz_uncompressed_gb
            self.sz_compressed_gb = self.vox_fused.sz_compressed_gb
        else:
            self.num_uncompressed_params = self.vox_rgb.num_uncompressed_params + self.vox_sigma.num_uncompressed_params
            self.num_compressed_params =  self.vox_rgb.num_compressed_params + self.vox_sigma.num_compressed_params
            self.sz_uncompressed_gb = self.vox_rgb.sz_uncompressed_gb + self.vox_sigma.sz_uncompressed_gb
            self.sz_compressed_gb = self.vox_rgb.sz_compressed_gb + self.vox_sigma.sz_compressed_gb
            
        self.compression_factor = self.num_uncompressed_params / self.num_compressed_params
        return {
            'num_uncompressed_params': self.num_uncompressed_params,
            'num_compressed_params': self.num_compressed_params,
            'sz_uncompressed_gb': self.sz_uncompressed_gb,
            'sz_compressed_gb': self.sz_compressed_gb,
            'compression_factor': self.compression_factor
        }
        