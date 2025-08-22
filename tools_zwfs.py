

import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import shutil
# mpl.rcParams['text.usetex'] = False  # ADIEU LaTeX
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False


from plot_functions.plot_func import *




def pad_to_square(arr):
    M, N = arr.shape
    diff = abs(N - M)
    
    if M < N:
        # Trop peu de lignes → on ajoute (diff) lignes
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        padded = np.pad(arr, pad_width=((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
        padded_cr = [-pad_bottom, 0,pad_top,0]
       
    elif N < M:
        # Trop peu de colonnes → on ajoute (diff) colonnes
        pad_left = diff // 2
        pad_right = diff - pad_left
        padded = np.pad(arr, pad_width=((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        padded_cr = [0,-pad_left,0, pad_right]
    else:
        # Déjà carré, miracle
        padded = arr
        padded_cr = 0
    
    return padded, padded_cr

def signal_processing_pupils(image, position_pupils, submasks):
    sub_images = []
    for j in range(2):
        minr, minc, maxr, maxc = position_pupils[j]
        sub_images.append(image[minr:maxr, minc:maxc])
    sub_images[0],_= pad_to_square(sub_images[0])
    sub_images[0]=sub_images[0]*submasks[0]#np.pad(pupilles[0], pad_width=((1, 1), (0, 0)), mode='constant', constant_values=0)
    sub_images[1],_= pad_to_square(sub_images[1])#np.pad(pupilles[1], pad_width=((1, 1), (0, 0)), mode='constant', constant_values=0)
    sub_images[1]=sub_images[1]*submasks[1]
    signals = []
    signals.append(sub_images[0][submasks[0]].ravel())
    
    signals.append(sub_images[1][submasks[1]].ravel())
    return signals, sub_images

def signal_processing_pupils_optimised(image, position_pupils, submasks):
   
    minr, minc, maxr, maxc = position_pupils
    signals = image[minr:maxr, minc:maxc][submasks]
    
    return signals


def save_dm_animation_as_gif(image_sequence, gif_filename='animated_discs.gif',
                                cmap='viridis', duration=0.3, disc_size=400,
                                dpi=100, add_colorbar=True, remove_temp=True):
    """
    Saves a 3D array of 2D images as a GIF using scatter plots with circular markers.
    
    Parameters:
        image_sequence (np.ndarray): Array of shape (num_frames, height, width).
        gif_filename (str): Output filename for the GIF.
        cmap (str): Colour map to apply to the disc intensities.
        duration (float): Duration (in seconds) of each frame in the GIF.
        disc_size (float): Area of the circular markers.
        dpi (int): Resolution of each saved frame.
        add_colorbar (bool): Whether to include a colour bar in each frame.
        remove_temp (bool): Whether to delete temporary image files after saving the GIF.
    """
    num_frames = image_sequence.shape[0]
    nrows, ncols = image_sequence.shape[1], image_sequence.shape[2]
    temp_dir = 'frames_temp'
    os.makedirs(temp_dir, exist_ok=True)

    for idx in range(num_frames):
        img = image_sequence[idx]
        x, y = np.meshgrid(np.arange(ncols), np.arange(nrows))
        x_flat = x.flatten()
        y_flat = y.flatten()
        values = img.flatten()
        
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(
            x_flat, y_flat,
            c=values,
            cmap=cmap,
            s=disc_size,
            edgecolors='none',
            marker='o'
        )
        
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-0.5, ncols - 0.5)
        ax.set_ylim(-0.5, nrows - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Frame {idx}", fontsize=12)
        
        if add_colorbar:
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
        
        plt.tight_layout()
        frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.png")
        plt.savefig(frame_path, dpi=dpi)
        plt.close()

    with imageio.get_writer(gif_filename, mode='I', duration=duration) as writer:
        for idx in range(num_frames):
            frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.png")
            image = imageio.imread(frame_path)
            writer.append_data(image)

    if remove_temp:
        shutil.rmtree(temp_dir)

    print(f"GIF saved as '{gif_filename}'.")

def save_raster_animation_as_gif(image_sequence, gif_filename='animated_raster.gif',
                                  cmap='viridis', duration=0.3, dpi=100,
                                  vmin=None, vmax=None,
                                  add_colorbar=True, remove_temp=True):
    """
    Saves a 3D array of 2D images as a GIF using imshow (standard square pixels).
    
    Parameters:
        image_sequence (np.ndarray): Array of shape (num_frames, height, width).
        gif_filename (str): Output filename for the GIF.
        cmap (str): Colour map to use for the image.
        duration (float): Duration (in seconds) of each frame in the GIF.
        dpi (int): Resolution of each saved frame.
        vmin (float): Minimum value for normalising colormap (optional).
        vmax (float): Maximum value for normalising colormap (optional).
        add_colorbar (bool): Whether to include a colour bar in each frame.
        remove_temp (bool): Whether to delete temporary files after saving the GIF.
    """
    num_frames = image_sequence.shape[0]
    temp_dir = 'frames_temp_raster'
    os.makedirs(temp_dir, exist_ok=True)

    for idx in range(num_frames):
        img = image_sequence[idx]
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(f"Frame {idx}", fontsize=12)
        ax.axis('off')

        if add_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')

        plt.tight_layout()
        frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.png")
        plt.savefig(frame_path, dpi=dpi)
        plt.close()

    with imageio.get_writer(gif_filename, mode='I', duration=duration) as writer:
        for idx in range(num_frames):
            frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.png")
            image = imageio.imread(frame_path)
            writer.append_data(image)

    if remove_temp:
        shutil.rmtree(temp_dir)

    print(f"GIF saved as '{gif_filename}'.")

def save_plot_animation_as_gif(data_sequence, gif_filename='animated_plot.gif',
                            duration=0.3, dpi=100,
                            x_values=None,
                            xlim=None, ylim=None,
                            line_kwargs=None,
                            xlabel=None, ylabel=None, title=None,
                            remove_temp=True):
    """
    Saves a sequence of 1D line plots as a GIF.
    
    Parameters:
        data_sequence (array-like): Sequence of 1D arrays (shape: num_frames × length).
        gif_filename (str): Path for the output GIF.
        duration (float): Duration per frame in seconds.
        dpi (int): Resolution of each frame.
        x_values (array-like): X-axis values (defaults to range(length)).
        xlim, ylim (tuple): Axis limits to fix during animation.
        line_kwargs (dict): Additional arguments passed to plt.plot().
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        title (str or callable): Static string or function `title(frame_idx)` for dynamic title.
        remove_temp (bool): Whether to delete temporary PNG files after the GIF is saved.
    """
    if isinstance(data_sequence, list):
        data_sequence = np.array(data_sequence)
    
    num_frames, data_length = data_sequence.shape
    if x_values is None:
        x_values = np.arange(data_length)
    if line_kwargs is None:
        line_kwargs = {}

    temp_dir = 'frames_temp_plot'
    os.makedirs(temp_dir, exist_ok=True)

    for idx in range(num_frames):
        y = data_sequence[idx]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_values, y, **line_kwargs)

        # Labels
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        # Title (static or dynamic)
        if callable(title):
            ax.set_title(title(idx), fontsize=12)
        elif isinstance(title, str):
            ax.set_title(title, fontsize=12)
        else:
            ax.set_title(f"Frame {idx}", fontsize=12)

        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)

        ax.grid(True)
        plt.tight_layout()
        frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.png")
        plt.savefig(frame_path, dpi=dpi)
        plt.close()

    with imageio.get_writer(gif_filename, mode='I', duration=duration) as writer:
        for idx in range(num_frames):
            frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.png")
            frame = imageio.imread(frame_path)
            writer.append_data(frame)

    if remove_temp:
        shutil.rmtree(temp_dir)

    print(f"GIF saved as '{gif_filename}'.")