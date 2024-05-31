import torch
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from opt import  yaml_config_parser, Config

import torch.nn as nn
from scipy.ndimage import rotate


from torchvision.transforms import functional as FF
import torch.nn.functional as F
from pytorch_msssim import ssim
import pywt

import torch
import torch.nn.functional as F
from math import ceil

from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None  # This disables the limit size

def get_kwargs_dict(config_file):
    args = Config()
    
    override_configs = yaml_config_parser(config_file)  # load specific configuration file
    base_config_file = override_configs.get('base_config', "")
    base_config_file = yaml_config_parser(base_config_file)

    #override base_config_file with the values from override_configs
    for key, value in override_configs.items():
        base_config_file[key] = value
    
    #override args - make sure that all arguments being parsed are in args
    for key, value in base_config_file.items():
        setattr(args, key, value)
    return args


def get_data_loader_for_current_reso(reso, batch_size=1024):
    all_coords = torch.cartesian_prod(torch.arange(reso), torch.arange(reso))# all combinations of x and y for resoxreso image
    train_loader = torch.utils.data.DataLoader(all_coords, batch_size=batch_size, shuffle=True)
    return train_loader

def mse2psnr(mse):
    # ensure x is a tensor
    if not isinstance(mse, torch.Tensor):
        mse = torch.Tensor([mse])
    # device should be same as mse
    mse2psnr_func = lambda x: -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))

    return mse2psnr_func(mse).item()

def psnr(original, compressed):
    # ensure they are same type
    #if not tensor make torch tensor
    if not isinstance(original, torch.Tensor):
        original = torch.Tensor(original)
    if not isinstance(compressed, torch.Tensor):
        compressed = torch.Tensor(compressed)

    mse = torch.mean((original - compressed) ** 2)
    return mse2psnr(mse)
    
def psnr_with_mask(img1, img2):
    """
    Computes the PSNR between two images over non-zero regions.
    Args:
        img1, img2: 2D or 3D PyTorch Tensors representing images.
    Returns:
        psnr: float, the computed PSNR value.
    """
    if img1.shape != img2.shape:
        raise ValueError("The shapes of the two images must be the same.")
        
    # Create a binary mask representing the non-zero (non-padding) regions of img1
    mask = img1 != 0
    
    # Compute the MSE over non-zero regions
    mse = torch.sum((img1[mask] - img2[mask]) ** 2) / torch.sum(mask)
    
    if mse == 0:
        return float('inf')  # Return infinity if the images are identical
    
    # Compute the PSNR
    return mse2psnr(torch.Tensor([mse]))




def load_target(args):
    if args.dimensions == 2:
        target = get_target_image(args.target, args.payload_position, normalization = "min_max", payload=args.payload, dtype=args.dtype)
    elif args.dimensions == 3:
        target = get_target_3d_object(args.target, args.payload_position, normalization = "min_max", payload=args.payload, dtype=args.dtype)
    else:
        raise ValueError("Invalid number of dimensions. Supported values are 2 and 3.")
    return target


def preprocess(data_path, payload_position="grayscale", dtype="float32", normalization="min_max", dimensions=2, payload=0):
    if dtype == "float32":
        dtype = torch.float32
    elif dtype == "float64":
        dtype = torch.float64
    else:
        raise NotImplementedError("dtype not implemented")
    
    print("data_path", data_path)
    if data_path.endswith(".npy"):
        data = np.load(data_path)
    elif data_path.endswith((".jpg", ".png", ".jpeg")):
        data = Image.open(data_path)
        # if it has 4 channels, remove the alpha channel
        if data.mode == 'RGBA':
            data = data.convert('RGB')
    elif data_path.endswith(".raw"):
        data = np.fromfile(data_path, dtype=np.uint8)
        if data.shape[0] == 64**3:
            data = data.reshape(64, 64, 64)
        elif data.shape[0] == 128**3:
            data = data.reshape(128, 128, 128)
        elif data.shape[0] == 256**3:
            data = data.reshape(256, 256, 256)
        elif data.shape[0] == 512**3:
            data = data.reshape(512, 512, 512)
        elif data.shape[0] == 1024**3:
            data = data.reshape(1024, 1024, 1024)
        elif data.shape[0] == 2048**3:
            data = data.reshape(2048, 2048, 2048)
        else:
            raise Exception("Invalid raw data shape")

    elif "cube" in data_path or "image" in data_path:
        dim_identifier = "cube" if "cube" in data_path else "image"
        target_dim = data_path.split(dim_identifier)[-1]

        if target_dim.isdigit():
            target_dim = int(target_dim)
        else:
            raise Exception("Invalid target dimension")

        if "cube" in data_path:
            data = np.zeros((target_dim, target_dim, target_dim))
            data[target_dim // 2, target_dim // 2, target_dim // 2] = 1
        else:
            data = np.zeros((target_dim, target_dim))
            data[target_dim // 2, target_dim // 2] = 1

    
    else:
        raise Exception("Unsupported file format")

    # Check if uint16
    if isinstance(data, np.ndarray) and data.dtype == np.uint16:
        data = data.astype(np.int16)
    if isinstance(data, np.ndarray) and data.dtype == np.uint8:
        data = data.astype(np.int8)

    if isinstance(data, np.ndarray) and dimensions == 2:
        data = Image.fromarray(data)
    elif isinstance(data, np.ndarray) and dimensions == 3:
        tensor = torch.from_numpy(data)
        if payload_position == "grayscale" and len(data.shape) == 4:
            tensor = tensor.mean(dim=-1)  # Convert 3D object to grayscale-like representation
        if payload_position != "grayscale":
            # Unsqueeze to add channel dimension
            tensor = tensor.unsqueeze(0)


    if dimensions == 2:
        transform = transforms.Compose([
            transforms.PILToTensor(),
        ])
        if payload_position == "grayscale" or payload < 2:
            transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Lambda(lambda x: x.squeeze(0)),
            ])
            
        if payload == 1:
            transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Grayscale(num_output_channels=1),
            ])
            
        tensor = transform(data)
    
    tensor = tensor.float()

    if normalization == "min_max":
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    elif normalization == "mean_std":
        tensor = (tensor - tensor.mean()) / tensor.std()

    if payload_position == "first_core" or payload_position == "last_core":
        if len(tensor.shape) == 3:
            tensor = tensor.permute(1, 2, 0)
        if len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 3, 1)
        
    return tensor




def get_target_image(im_string, payload_position="first_core", dtype="float32", normalization="min_max", payload=0):
    return preprocess(im_string, payload_position, dtype, normalization, dimensions=2, payload=payload)

def get_target_3d_object(obj_string, payload_position="grayscale", dtype="float32", normalization="min_max", payload=0):
    return preprocess(obj_string, payload_position, dtype, normalization, dimensions=3, payload=payload)



def save_img(img, path, cmap=None):
    img = img.detach().cpu().numpy()
    if img.ndim == 3 and 3 not in img.shape:
        img = img[0] # hack for 3d objects
    if cmap:
        plt.imsave(path, img, cmap=cmap)
    else:
        plt.imsave(path, img)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
class SimpleSamplerImplicit:
    def __init__(self, dimensions, batch_size, max_value):
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_value = max_value

    def next_batch(self):
        # Generate random coordinates within the desired range [0, max_value]
        batch_indices = torch.randint(0, self.max_value + 1, size=(self.batch_size, self.dimensions))
        # Normalize coordinates to the range [-1, 1]
        normalized_indices = (batch_indices.float() / self.max_value) * 2 - 1
        
        return batch_indices, normalized_indices
    
class SimpleSamplerSubset:
    def __init__(self,dimensions,batch_size,max_value, indices):
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_value = max_value
        self.indices = indices

    def next_batch(self):
        # get random sample from indices
        batch_indices = self.indices[np.random.randint(0, len(self.indices), size=self.batch_size)]
        # Normalize coordinates to the range [-1, 1]
        normalized_indices = (batch_indices.float() / self.max_value) * 2 - 1

        return batch_indices, normalized_indices



class SimpleSamplerNonRandom:
    # used for testing - iterate over all samples once
    def __init__(self, dimensions, batch_size, max_value=32):
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_value = max_value
        self.total_samples = (max_value + 1) ** dimensions
        self.index = 0

    def next_batch(self):
        if self.index >= self.total_samples:
            self.index = 0
            print("Resetting sampler index")

        num_samples = min(self.batch_size, self.total_samples - self.index)
        indices = np.arange(self.index, self.index + num_samples)

        if self.dimensions == 2:
            x = indices % (self.max_value + 1)
            y = indices // (self.max_value + 1)
            batch_indices = np.column_stack((x, y))
        elif self.dimensions == 3:
            x = indices % (self.max_value + 1)
            yz = indices // (self.max_value + 1)
            y = yz % (self.max_value + 1)
            z = yz // (self.max_value + 1)
            batch_indices = np.column_stack((x, y, z))
        else:
            raise ValueError("Unsupported number of dimensions")

        self.index += num_samples

        # Normalize coordinates to the range [-1, 1]
        normalized_indices = torch.tensor((batch_indices.astype(np.float32) / self.max_value) * 2 - 1)

        return torch.tensor(batch_indices), normalized_indices


def calculate_gamma(lr0, decay_factor, num_iters):
    lrT = lr0 * decay_factor
    gamma = (lrT / lr0) ** (1 / num_iters)
    return gamma


def linear_warmup_lr_scheduler(optimizer, warmup_steps):
    def lr_lambda(current_step):
        return min(1.0, float(current_step + 1) / warmup_steps)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def setup_rank_upsampling(model, rank_range, iteration_range=(50, 1000)):
    """
    Sets up rank upsampling for a model.

    Parameters:
    - model: The model to set up rank upsampling for.
    - rank_range: Tuple or list with three elements (start_rank, end_rank, step) defining the range of mask ranks.
    - iteration_range: Tuple with two elements (start_iteration, end_iteration) defining the range of upsampling iterations.
    """
    start_rank, end_rank, step = rank_range
    start_ite, end_ite = iteration_range

    new_max_ranks = list(range(start_rank, end_rank, step))
    
    if len(new_max_ranks) > 1:
        model.mask_rank = start_rank - step
        model.create_rank_upsampling_mask()
        
        step_size = (end_ite - start_ite) // (len(new_max_ranks) - 1)
        new_rank_upsample_iterations = list(range(start_ite, end_ite, step_size))

        print("New ranks:", new_max_ranks)
        print("New rank upsample iterations:", new_rank_upsample_iterations)
    else:  
        new_rank_upsample_iterations = []
        new_max_ranks = []
        
        print("Error: The rank range provided does not generate a valid rank list. Please check the inputs.")
        
    return new_max_ranks, new_rank_upsample_iterations


def make_rotation_of_target(target, degrees, num_dims, zero_padding=True):
    """
    Args:
        target: tensor of shape 2D or 3D
        degrees: int
        num_dims: int
    Returns:
        rotated_target: tensor of shape 2D or 3D
    """
    # Ensure target is a PyTorch Tensor.
    if not isinstance(target, torch.Tensor):
        raise TypeError("target should be a PyTorch Tensor")

    # Check the number of dimensions in the target tensor.
    dims = len(target.shape)
    if dims != num_dims:
        raise ValueError("Dimensions of target do not match specified num_dims")

    # Zero Padding
    if zero_padding:
        h, w = target.shape[0], target.shape[1]
        new_dim_h, new_dim_w = h * 2, w * 2
        offset_h, offset_w = h // 2, w // 2
        if dims == 2:
            zeros = torch.zeros(new_dim_h, new_dim_w).float()
            zeros[offset_h:offset_h + h, offset_w:offset_w + w] = target
        elif dims == 3:
            c = target.shape[2]
            new_dim_c, offset_c = c * 2, c // 2
            zeros = torch.zeros(new_dim_h, new_dim_w, new_dim_c).float()
            zeros[offset_h:offset_h + h, offset_w:offset_w + w, offset_c:offset_c + c] = target
        target = zeros

    # Rotate the target tensor.
    if dims == 2:
        target = target.unsqueeze(0)  # Shape: [1, H, W]
        rotated_target = FF.rotate(target, degrees)
        rotated_target = rotated_target.squeeze(0)  # Shape: [H, W]
    elif dims == 3:
        #target = target.permute(2, 0, 1)  # Shape: [C, H, W]
        target_np = target.cpu().numpy()
        rotated_target_np = rotate(target_np, degrees, axes=(1,2), reshape=False)
        rotated_target = torch.tensor(rotated_target_np).float()
        
    return rotated_target


def make_noisy_target(target, args):
    """
    Args:
        target: tensor of shape 2D or 3D
        args: args from command line that contain noise_type, noise_std, noise_mean
    Returns:
        noisy_target: tensor of shape 2D or 3D
    """
    noise_type = args.noise_type
    noise_std = args.noise_std
    noise_mean = args.noise_mean

    if noise_type == "gaussian":
        noise = torch.randn_like(target) * noise_std + noise_mean
    elif noise_type == "salt_and_pepper":
        noise = torch.rand_like(target)
        noise[noise < noise_mean] = 0
        noise[noise > 1 - noise_mean] = 1
    elif noise_type == "laplace":
        noise = torch.distributions.laplace.Laplace(0, noise_std).sample(target.shape)
    else:
        raise ValueError("Invalid noise type")

    noisy_target = target + noise
    noisy_target = torch.clamp(noisy_target, 0, 1)

    dtype = args.dtype
    if dtype == "float32":
        noisy_target = noisy_target.float()
    elif dtype == "float64":
        noisy_target = noisy_target.double()
    else:
        raise NotImplementedError("dtype not implemented")

    return noisy_target



@torch.no_grad()
def calculate_ssim(original, compressed, payload, dim, patched=False):
    # Add batch dimension and move to CPU
    original = original.unsqueeze(0).cpu()
    compressed = compressed.unsqueeze(0).cpu()

    if payload == 0:
        original = original.unsqueeze(-1)
        compressed = compressed.unsqueeze(-1)

    # Adjust channel/depth dimension for 2D or 3D data
    if dim == 2:  
        original = original.permute(0, 3, 1, 2)
        compressed = compressed.permute(0, 3, 1, 2)
    if dim == 3: # we want B x C x D x H x W
        # loop over each W)
        if payload == 0:
            original = original.permute(0, 4, 1, 2, 3)
            compressed = compressed.permute(0, 4, 1, 2, 3)
        else:
            compressed = compressed.permute(0, 4, 1, 2, 3)

        ssim_values = []
        #for i in range(original.shape[1]):
        # make tqdm
        for i in tqdm(range(original.shape[2]), desc='Processing SSIM'):
            ssim_values.append(ssim(original[:,:,i], compressed[:,:,i], data_range=1.0).item())
        return np.mean(ssim_values)

    if patched:
        return ssim_patchwise(original, compressed, window_size=51, size_average=True, full=False, data_range=None).item()
    return ssim(original, compressed, data_range=1.0).item()



def ssim_patchwise(target, recon, window_size=11, size_average=True, full=False, data_range=None):
    C1 = 0.01 ** 2  # constant for luminance comparison
    C2 = 0.03 ** 2  # constant for contrast and structure comparison
    
    target = target.detach().cpu()
    recon = recon.detach().cpu()
    
    (_, channels, height, width) = recon.shape
    
    ssim_vals = []
    stride = window_size
    nh = ceil(height / stride)
    nw = ceil(width / stride)
    
    for h in range(0, height, stride):
        for w in range(0, width, stride):
            end_h = min(h + stride, height)
            end_w = min(w + stride, width)
            
            patch_t = target[:, :, h:end_h, w:end_w]
            patch_r = recon[:, :, h:end_h, w:end_w]
            
            # Padding the patches if necessary
            pad_h = stride - (end_h - h)
            pad_w = stride - (end_w - w)
            
            if pad_h > 0 or pad_w > 0:
                padding = (0, pad_w, 0, pad_h)
                patch_t = F.pad(patch_t, padding)
                patch_r = F.pad(patch_r, padding)
            
            mu1 = patch_t.mean(dim=[2, 3], keepdim=True)
            mu2 = patch_r.mean(dim=[2, 3], keepdim=True)
            
            sigma1_sq = F.mse_loss(patch_t, mu1, reduction='none').mean(dim=[2, 3], keepdim=True)
            sigma2_sq = F.mse_loss(patch_r, mu2, reduction='none').mean(dim=[2, 3], keepdim=True)
            sigma12 = (patch_t * patch_r).mean(dim=[2, 3], keepdim=True) - mu1 * mu2
            
            ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
            ssim_vals.append(ssim_map.mean(dim=[1, 2, 3]))
            
    ssim_vals = torch.stack(ssim_vals)
    
    if size_average:
        return ssim_vals.mean()
    else:
        return ssim_vals.mean(1)



def check_validity_upsampling_steps(init_res, end_res, upsampling_iterations, num_iterations):
    """
    This function checks whether the given upsampling steps are valid based on
    the initial and end resolutions, upsampling iterations, and number of iterations.
    
    :param init_res: Initial resolution.
    :param end_res: End resolution.
    :param upsampling_iterations: List of iterations at which upsampling occurs.
    :param num_iterations: Total number of iterations.
    :return: Boolean indicating whether the upsampling steps are valid.
    # # Example Usage:
    # init_res = 32
    # end_res = 256
    # upsampling_iterations = [64, 128]
    # num_iterations = 500

    # is_valid = check_validity_upsampling_steps(init_res, end_res, upsampling_iterations, num_iterations)
    """
    if init_res == end_res:
        if len(upsampling_iterations) == 0 or len(upsampling_iterations) == 1 and upsampling_iterations[0] > num_iterations:
            return True

    # Check if init_res is less than end_res
    if init_res >= end_res:
        print(f"Invalid: init_res ({init_res}) should be less than end_res ({end_res})")
        return False
    
    # Check if the last element in upsampling_iterations is less than num_iterations
    if upsampling_iterations and upsampling_iterations[-1] >= num_iterations:
        print(f"Invalid: last element in upsampling_iterations ({upsampling_iterations[-1]}) should be less than num_iterations ({num_iterations})")
        return False
    
    # Check length of upsampling_iterations is matching number of doublings between init_res and end_res
    num_doublings = int(np.log2(end_res) - np.log2(init_res))
    if len(upsampling_iterations) != num_doublings:
        print(f"num_doublings:, upsampling_iterations, len(upsampling_iterations), end_res, init_res", num_doublings, upsampling_iterations, len(upsampling_iterations), end_res, init_res)
        return False
    
    # If all checks passed, return True
    print("Upsampling steps are valid.")
    return True


def custom_masked_avg_pool2d(input, kernel_size, stride):
        """
        Example: consider a 2x2 window with the values [2, 2, 0, 0].
        With regular average pooling, the output would be (2+2+0+0)/4 = 1.
        With the masked average pooling (where zeros are ignored), the output would be (2+2)/2 = 2.
        """
        # Create a mask of non-zero values
        mask = (input != 0).float()
        input_sum = torch.nn.functional.avg_pool2d(input, kernel_size, stride)
        mask_sum = torch.nn.functional.avg_pool2d(mask, kernel_size, stride)
        return input_sum / (mask_sum + 1e-10)

def custom_masked_avg_pool3d(input, kernel_size, stride):
        # Create a mask of non-zero values
        mask = (input != 0).float()
        input_sum = torch.nn.functional.avg_pool3d(input, kernel_size, stride)
        mask_sum = torch.nn.functional.avg_pool3d(mask, kernel_size, stride)
        return input_sum / (mask_sum + 1e-10)

def downsample_with_avg_pooling(target_tmp, factor, dim, grayscale=0, device=None, masked=False):

    if dim == 1:
        avg_pool = torch.nn.AvgPool1d(kernel_size=factor, stride=factor)
        downsampled_target = avg_pool(target_tmp.unsqueeze(0)).squeeze(0)
    elif dim == 2:
        if masked:
            downsample_func = custom_masked_avg_pool2d
        else:
            downsample_func = torch.nn.functional.avg_pool2d

        if grayscale:
            target_tmp = torch.stack([target_tmp for i in range(3)], dim=0)
            downsampled_target = downsample_func(target_tmp, kernel_size=factor, stride=factor)
            downsampled_target = downsampled_target[0].to(device)
        else:
            target_tmp = target_tmp.permute(2, 0, 1)
            downsampled_target = downsample_func(target_tmp, kernel_size=factor, stride=factor)
            downsampled_target = downsampled_target.permute(1, 2, 0).to(device)
    elif dim == 3:
        # NOTE: Masked average pooling for 3D not implemented here for simplicity.
        avg_pool = torch.nn.AvgPool3d(kernel_size=factor, stride=factor)
        if grayscale:
            target_tmp = torch.stack([target_tmp for i in range(3)], dim=0).unsqueeze(0)
            downsampled_target = avg_pool(target_tmp).squeeze(0)
            downsampled_target = downsampled_target[0].to(device)
        else:
            if len(target_tmp.shape) == 3:
                target_tmp = target_tmp.permute(2, 0, 1).unsqueeze(0)
                downsampled_target = avg_pool(target_tmp).squeeze(0)
                downsampled_target = downsampled_target.permute(1, 2, 0).to(device)
            elif len(target_tmp.shape) == 4:
                downsampled_target = avg_pool(target_tmp)
                downsampled_target = downsampled_target.permute(1, 2, 3, 0)
    else:
        raise ValueError(f"Invalid value for dim parameter. Expected 1, 2, or 3, but got {dim}")

    return downsampled_target


def downsample_with_lanczos( target, factor, dim):
    if dim not in [1, 2]:
        raise ValueError("Only dim 1 and 2 are supported for this Lanczos downsampling function.")

    # Convert torch tensor to PIL Image
    if dim == 1:
        raise ValueError("Lanczos downsampling for 1D data is not typically supported in imaging libraries.")
    elif dim == 2:
        # Convert the tensor to the range [0, 255] and to 'uint8' type
        target_uint8 = (target * 255).byte().numpy()
        img = Image.fromarray(target_uint8)

        # Compute new dimensions
        new_dims = (img.width // factor, img.height // factor)
        
        # Downsample using Lanczos
        img_downsampled = img.resize(new_dims, Image.LANCZOS)

        # Convert back to torch tensor and normalize to [0, 1]
        target_downsampled = torch.from_numpy(np.array(img_downsampled)).float() / 255.0

    return target_downsampled

def downsample_with_wavelet( target, factor, dim):
    if dim not in [1, 2]:
        raise ValueError("Only dim 1 and 2 are supported for this wavelet downsampling function.")
    input_shape = target.shape
    did_squeeze = False
    if len(input_shape) == 3 and 1 in input_shape:
        target = target.squeeze()
        did_squeeze = True

    for _ in range(int(np.log2(factor))):
        if dim == 1:
            coeffs = pywt.dwt(target.numpy(), 'haar')
            target = torch.from_numpy(coeffs[0])  # Use approximate coefficients
        elif dim == 2:
            coeffs = pywt.dwt2(target.numpy(), 'haar')
            target = torch.from_numpy(coeffs[0])  # Use LL coefficients (approximation)

    # get the value ranges
    min_val = target.min()
    max_val = target.max()
    
    # normalize between 0 and 1
    target = (target - min_val) / (max_val - min_val)

    if did_squeeze:
        target = target.unsqueeze(-1)

    return target



def get_model_args(args, target, noisy_target, device_type):
    # Default parameters from BaseTNModel
    model_args = {
        'max_rank': args.max_rank,
        'dtype': args.dtype,
        'loss_fn_str': args.loss_fn_str, 
        'use_TTNF_sampling': args.use_TTNF_sampling,
        'payload': args.payload,
        'payload_position': args.payload_position, 
        'canonization': args.canonization,
        'activation': args.activation,
        'compression_alg':  args.compression_alg, 
        'regularization_type': args.regularization_type,
        'dimensions': args.dimensions,
        'regularization_weight': args.regularization_weight,
        'masked_avg_pooling': args.masked_avg_pooling,
        'init_reso' : args.init_reso,
        "target": target,
        "noisy_target": noisy_target,
        "device": device_type,
    }
    
    if args.model == "TT":
        model_args.update({
        "is_tensor_ring": args.is_tensor_ring,
        'channel_rank': args.channel_rank,
        'sigma_init': args.sigma_init,
        })
        
    if args.model == "QTT":
        model_args.update({
        'sigma_init': args.sigma_init,
        })
        
    
    if args.model == "QTT-MLP":
        tt_mlp_specific = {
        'shadingMode': args.shading_mode,
        'output_dim': args.output_dim,
        'fea_pe': args.fea_pe,
        'featureC': args.featureC
        }
        # add tt_mlp_specific to model_args
        model_args.update(tt_mlp_specific)

    return model_args


def calculate_iterations_until_next_upsampling(args, iterations):
    iterations_for_upsampling = args.iterations_for_upsampling
    if len(iterations_for_upsampling) > 0:
        iterations_until_next_upsampling = [
            iterations_for_upsampling[i + 1] - iterations_for_upsampling[i] for i in range(len(iterations_for_upsampling) - 1)
        ]
        iterations_until_next_upsampling.append(iterations - iterations_for_upsampling[-1])
    else:
        # Handle the case when 'iterations_for_upsampling' is empty
        iterations_until_next_upsampling = [iterations+1]
        iterations_for_upsampling = [iterations+1]
    return iterations_for_upsampling,iterations_until_next_upsampling



