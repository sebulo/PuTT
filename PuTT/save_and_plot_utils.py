import os
import datetime
import torch
import matplotlib.pyplot as plt
from utils import *
import wandb
import numpy as np
import time



def plot_loss_and_saved_images(use_wandb, losses, saved_images, saved_images_iterations, psnr_arr, figsize, save_times, gt_list):
    if not use_wandb:
        plot_loss(losses, figsize=figsize)
        if saved_images:   
            plot_saved_img_iterations(saved_images, saved_images_iterations, psnrs=psnr_arr, figsize=(20,20), max_images_per_row=4, save_times=save_times, gt_list=gt_list)

def plot_loss(losses, figsize=(10,5), title="Losses"):
    # plot next to each other
    print("Fig size", figsize)
    plt.figure( figsize=figsize)
    plt.plot(losses)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def save_results_locally(args, saved_images, saved_images_iterations, save_times, psnr_arr, gt_list, best_recon, model):
    if not args.use_wandb and args.save_end_results_locally:
        save_string = generate_save_string(args)
        folder_path = os.path.join(args.save_dir, save_string)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        saved_ims_plot = save_all_data(saved_images, saved_images_iterations, save_times, psnr_arr, folder_path, args, gt_list=gt_list)
        save_imgs_and_plots(args, saved_ims_plot, best_recon, model, folder_path)


def generate_save_string(args):
    save_string = f"_num_iterations{args.num_iterations}_iterations_for_upsampling{args.iterations_for_upsampling}_init_reso_{args.init_reso}_max_rank_{args.max_rank}_max_batch_size_{args.max_batch_size}_lr{args.lr}_payload_{args.payload}"
    save_string = f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" + save_string
    save_string = f"{args.target}" + save_string
    return save_string


def save_imgs_and_plots(args, saved_ims_plot, best_recon, model, folder_path):
    saved_ims_plot.savefig(os.path.join(folder_path, "saved_ims_plot.png"))
    save_img(model.target, os.path.join(folder_path, "target.png"), cmap="gray" if not model.grayscale else None)
    save_img(best_recon, os.path.join(folder_path, "best_recon.png"), cmap="gray" if not model.grayscale else None)
    print("Best recon range: " + str(torch.min(best_recon)) + " to " + str(torch.max(best_recon)))
    print("target range: " + str(torch.min(model.target)) + " to " + str(torch.max(model.target)))
    best_recon = (best_recon - torch.min(best_recon)) / (torch.max(best_recon) - torch.min(best_recon))


def plot_psnrs(args, psnr_arr, figsize):
    if not args.use_wandb:
        plt.figure(figsize=figsize)
        plt.plot(psnr_arr)
        plt.title("PSNR")
        plt.show()


def save_data_while_training(args, model, figsize, saved_images, saved_images_iterations, save_times, time_start, psnr_arr, iteration):
    should_plot = False # figure out if we should plot'
    if args.plot_upsampling:
        print("Plotting at iteration UPS" + str(iteration))
        should_plot = True
    elif args.plot_training and iteration % args.save_every == 0 and iteration > args.save_every_start_iteration:
        print("Plotting at iteration TR" + str(iteration))
        should_plot = True
    
    image = model.get_image_plot(figsize=figsize, show_img = should_plot)
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if args.dimensions == 3 and len(image.shape) > 2:
        image = image[int(image.shape[0]/2)] 

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    saved_images.append(image)
    saved_images_iterations.append(iteration-1)
    save_times.append(time.time() - time_start)
    psnr_val = psnr(model.downsampled_target.detach().cpu(), image)
    psnr_arr.append(psnr_val)
    

def should_skip_saving(model, args):
    is_tt_model_with_many_factors = model.model == "QTT" and len(model.shape_factors) > 25
    is_high_reso_3D_model = model.current_reso > 1000 and model.dimensions == 3 and model.model in ["CP", "VM"]
    
    return not args.calculate_psnr_while_training or is_tt_model_with_many_factors or is_high_reso_3D_model


def save_image_to_wandb(model, best_recon, ite, save_locally=True, PSNR = None, SSIM=None):
    best_recon = best_recon.detach().cpu().numpy()
    
    if model.dimensions == 2 or model.dimensions == 3:
        best_recon_normalized  = (best_recon - np.min(best_recon)) / (np.max(best_recon) - np.min(best_recon))
        if model.dimensions == 3:
            best_recon_normalized = best_recon_normalized[int(best_recon_normalized.shape[0]/2)]
        image_name = f"Image_iteration_{ite}"
        best_recon_normalized = best_recon_normalized.squeeze()

        # check if PSNR and SSIM are tensors then item()
        if PSNR is not None and isinstance(PSNR, torch.Tensor):
            PSNR = PSNR.item()
        if SSIM is not None and isinstance(SSIM, torch.Tensor):
            SSIM = SSIM.item()
        if PSNR is not None:
            image_name += f"_psnr_{round(PSNR, 3)}"
        if SSIM is not None:
            image_name += f"_ssim_{round(SSIM, 3)}"
        image_name += ".png"

        if save_locally:
            filename = os.path.join(wandb.run.dir, image_name)
            print("Saving image to " + filename)
            plt.imsave(filename, best_recon_normalized, cmap='gray')
        else:
            wandb.log({image_name: wandb.Image(best_recon_normalized)}, step=ite+1)
    else:
        print("Not 2D or 3D - not saving to wandb")



def normalize_best_recon(best_recon):
    return (best_recon - np.min(best_recon)) / (np.max(best_recon) - np.min(best_recon))

def calculate_and_log_psnr(model, best_recon, noisy_target, use_wandb, args):
    gt_list = [model.target.numpy()]
    if args.noise_std > 0 and args.noise_type != "None" and args.get_noisy_target_psnr_measures:
        gt_list.append(noisy_target.numpy())
        psnr_noisy = psnr(model.target, noisy_target)
        print("PSNR of Taget and Noisy Target: " + str(psnr_noisy))
        
        psnr_recon_noisy = psnr(best_recon, noisy_target)
        print("PSNR of Recon and Noisy Target: " + str(psnr_recon_noisy))
        if use_wandb:
            wandb.log({"PSNR_noisy_vs_target": psnr_noisy})
            wandb.log({"PSNR_recon_vs_noisy": psnr_recon_noisy})
    return gt_list


def save_reconstruction(args, best_recon):
    best_recon_np = None
    slice_ = int(best_recon.shape[0]/2)

        # Check if best_recon is a PyTorch tensor and if it's on GPU
    if best_recon is not None:
        if torch.is_tensor(best_recon):
            if best_recon.is_cuda:
                best_recon_np = best_recon.detach().cpu().numpy()
            else:
                best_recon_np = best_recon.detach().numpy()
        elif isinstance(best_recon, np.ndarray):
            best_recon_np = best_recon  # best_recon is already a numpy array

        # Check if best_recon_np was successfully created or assigned
    if best_recon_np is not None:
            # Construct the file path
        file_path = f"{args.target}_{args.model}_{args.target}_{args.max_rank}__{slice_}{args.max_rank}__{slice_}{args.init_reso}_{args.end_reso}_{args.num_iterations}_{args.seed}.npy"

        # Save as npy file
        np.save(file_path, best_recon_np)



def plot_saved_img_iterations(saved_images, saved_images_iterations, psnrs=None, figsize=(10, 5), max_images_per_row=5, save_times=None, losses=None, gt_list=[]):
    num_images = len(saved_images)
    num_images += len(gt_list) # adds room for ground truth images
    num_rows = int(np.ceil(num_images / max_images_per_row))
    fig, axs = plt.subplots(num_rows, max_images_per_row, figsize=figsize)
    num_images -= len(gt_list)  # subtract room for ground truth images for loop object
    axs = axs.flatten()
    last_iteration = -1

    # check if all dims are the same in shape of one of the saved images
    # check that saved_images[0].shape[0] == saved_images[0].shape[1] == saved_images[0].shape[2]
    has_payload = False
    if saved_images[0].ndim == 3:
        has_payload = not saved_images[0].shape[0] == saved_images[0].shape[1] == saved_images[0].shape[2]
    if saved_images[0].ndim == 4:
        has_payload = not saved_images[0].shape[0] == saved_images[0].shape[1] == saved_images[0].shape[2] == saved_images[0].shape[3]

    for i in range(num_images):
        if saved_images[i].ndim == 2:  # For grayscale 2D images
            axs[i].imshow(saved_images[i], cmap="gray")
        elif saved_images[i].ndim == 3 and has_payload and 1 in saved_images[i].shape:
            axs[i].imshow(saved_images[i], cmap="gray")
        elif saved_images[i].ndim == 3 and has_payload: 
            axs[i].imshow(saved_images[i])
        elif saved_images[i].ndim == 3:  # For 3D structures
            middle_slice = saved_images[i].shape[0] // 2
            #axs[i].imshow(saved_images[i][:,:,middle_slice], cmap="gray")
            axs[i].imshow(saved_images[i][:,middle_slice,:], cmap="gray")
            #axs[i].imshow(saved_images[i][middle_slice,:,:], cmap="gray")
        elif saved_images[i].ndim == 4 and has_payload:  # For 3D structures with payload
            middle_slice = saved_images[i].shape[0] // 2
            axs[i].imshow(saved_images[i][middle_slice], cmap="gray")

        
        title = ""
        if saved_images_iterations[i] - last_iteration < 2:
            title = f"upsample {saved_images_iterations[i]}"
            last_iteration = -1
        else:
            title = f"iteration {saved_images_iterations[i]}"
            last_iteration = saved_images_iterations[i]
        if save_times and len(save_times) > 0:
            title += f" time {np.round(save_times[i], 1)}"
        if losses:
            title += f" loss {np.round(losses[i], 5)}"
        if psnrs:
            title += f" psnr {np.round(psnrs[i], 2)}"
        axs[i].set_title(title)
        axs[i].axis("off")
        
    for index, gt in enumerate(gt_list):
        print("gt.shape", gt.shape)
        axis_index = -1 - index
        middle_slice = gt.shape[0] // 2
        if gt.ndim == 2:  # For grayscale 2D images
            axs[axis_index].imshow(gt, cmap="gray")
        elif gt.ndim == 3 and has_payload: 
            if 1 in gt.shape:
                axs[axis_index].imshow(gt, cmap="gray")
            else:
                axs[axis_index].imshow(gt)
        elif gt.ndim == 3:  # For 3D structures
            axs[axis_index].imshow(gt[:, middle_slice], cmap="gray")
        elif gt.ndim == 4:  # For 3D structures with payload
            if 1 in gt.shape:
                if gt.shape[0] ==1:
                    # permute with numpy to get middle slice
                    gt = torch.from_numpy(gt).permute(1, 2, 3,0).numpy() # TODO still not functioning correctly
                axs[axis_index].imshow(gt[middle_slice], cmap="gray")
            else:
                axs[axis_index].imshow(gt[middle_slice])

        if index == 0:
            axs[axis_index].set_title("ground truth")
        else: 
            axs[axis_index].set_title(f"noisy ground truth")

        
    plt.tight_layout()
    plt.show()
    return fig

def save_all_data(saved_images, saved_images_iterations, save_times, psnr_arr, save_folder, args, gt=None):
    
    # save images 
    save_path = os.path.join(save_folder, "saved_images.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(saved_images, f)

    # save iterations
    save_path = os.path.join(save_folder, "saved_images_iterations.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(saved_images_iterations, f)

    # save save times
    save_path = os.path.join(save_folder, "save_times.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(save_times, f)

    # save psnr
    save_path = os.path.join(save_folder, "psnr.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(psnr_arr, f)

    # save config "args" to json in folder
    save_path = os.path.join(save_folder, "config.json")
    # save with pickle
    with open(save_path, 'w') as f:
        json.dump(make_dict_from_args(args), f)

    print("len(saved_images) ", len(saved_images))
    if len(saved_images) > 0:
        saved_ims_plot = plot_saved_img_iterations(saved_images, saved_images_iterations, psnrs=psnr_arr, figsize=(20,20), max_images_per_row=4, save_times=save_times, losses=None, gt=gt)

    return saved_ims_plot

def plot_image( image, title = "",size = (10,10)):
    with torch.no_grad():
        image = image.cpu().numpy()
        plt.figure(figsize=size)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()

def plot3dslices(targets,best_slices, figsize, title, cmap = "gray"):
    """
    Plots 3D slices (axial, coronal, and sagittal) and optionally saves them.

    :param slices: List containing the three slices to be plotted.
    :param figsize: Tuple representing the size of the figure.
    :param title: Title of the plot.
    :param save_locally: Boolean to decide whether to save the plot.
    :param save_path: Path where the plot should be saved.
    """
    if not best_slices or len(best_slices) != 3:
        print("Invalid input: 'slices' must be a list of three elements.")
        return

    plt.figure(figsize=figsize)

    slice_names = ['Axial', 'Coronal', 'Sagittal']
    for i, slice_ in enumerate(best_slices):
        plt.subplot(2, 3, i+1)
        plt.imshow(slice_, cmap=cmap)
        plt.title(f"recon - {slice_names[i]}")
        plt.subplot(2, 3, i+4)
        plt.imshow(targets[i], cmap=cmap)
        plt.title(f"target - {slice_names[i]}")
        plt.axis('off')

    plt.tight_layout()