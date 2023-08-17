"""
Provides some auxiliary functions for the main module
"""


# Import statements
import torch
import numpy as np
import skimage
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as visionF


# Function definitions
def landmark_coordinates(maps, device):
    """
    Generate the center coordinates as tensor for the current net.
    Parameters
    ----------
    maps: torch.utils.data.DataLoader
        Data loader for the current data split.
    device: string
        Net that generates assignment maps.
    Returns
    ----------
    loc_x: Tensor
        The centroid x coordinates
    loc_y: Tensor
        The centroid y coordinates
    grid_x: Tensor
    grid_y: Tensor
    """
    grid_x, grid_y = torch.meshgrid(torch.arange(maps.shape[2]),
                                    torch.arange(maps.shape[3]))
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(device)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(device)

    map_sums = maps.sum(3).sum(2).detach()
    maps_x = grid_x * maps
    maps_y = grid_y * maps
    loc_x = maps_x.sum(3).sum(2) / map_sums
    loc_y = maps_y.sum(3).sum(2) / map_sums

    return loc_x, loc_y, grid_x, grid_y

def landmarks_to_rgb(maps):
    """
    Converts the attention maps to maps of colors
    Parameters
    ----------
    maps: Tensor, [number of parts, width_map, height_map]
        The attention maps to display

    Returns
    ----------
    rgb: Tensor, [width_map, height_map, 3]
        The color maps
    """
    colors = [[0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25],
    [0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25],
    [0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25]]
    rgb = np.zeros((maps.shape[1],maps.shape[2],3))
    for m in range(maps.shape[0]):
        for c in range(3):
            rgb[:,:,c] += maps[m,:,:]*colors[m][c]
    return rgb

def rigid_transform(img, angle, translate, scale, invert=False):
    """
    Affine transforms input image
    Parameters
    ----------
    img: Tensor
        Input image
    angle: int
        Rotation angle between -180 and 180 degrees
    translate: [int]
        Sequence of horizontal/vertical translations
    scale: float
        How to scale the image
    invert: bool
        Whether to invert the transformation

    Returns
    ----------
    img: Tensor
        Transformed image
    """
    shear = 0
    bilinear = visionF.InterpolationMode.BILINEAR
    if not invert:
        img = visionF.affine(img, angle, translate, scale, shear,
                             interpolation=bilinear)
    else:
        translate = [-t for t in translate]
        img = visionF.affine(img, 0, translate, 1, shear)
        img = visionF.affine(img, -angle, [0,0], 1/scale, shear)
    return img

def show_maps(ims,maps,loc_x,loc_y, epoch, experiment, savefig=False):
    """
    Plot images, attention maps and landmark centroids.
    Parameters
    ----------
    ims: Tensor, [batch_size, 3, width_im, height_im]
        Input images on which to show the attention maps
    maps: Tensor, [batch_size, number of parts, width_map, height_map]
        The attention maps to display
    loc_x: Tensor, [batch_size, 0, number of parts]
        The centroid x coordinates
    loc_y: Tensor, [batch_size, 0, number of parts]
        The centroid y coordinates
    savefig: bool
        If true, save the figures to file, otherwise simply show the plots
    """
    colors = [[0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25],
    [0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25],
    [0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25]]
    fig,axs = plt.subplots(3,3)
    i = 0
    for ax in axs.reshape(-1):
        if i<maps.shape[0]:
            landmarks = landmarks_to_rgb( maps[i,0:-1,:,:].detach().cpu().numpy())
            ax.imshow((skimage.transform.resize(landmarks, (256, 256)) + skimage.transform.resize((ims[i, :, :, :].permute(1, 2, 0).numpy()), (256, 256))))
            ax.scatter(loc_y[i,0:-1].detach().cpu()*256/maps.shape[-1],loc_x[i,0:-1].detach().cpu()*256/maps.shape[-1],c=colors[0:loc_x.shape[1]-1],marker='x')
        i += 1

    if savefig==False:
        plt.show()
    else:
        plt.savefig(f'../results_{experiment}/{epoch}_{np.random.randint(0, 10)}')

def get_epoch(experiment):
    """
    Return the last epoch saved by the model
    Parameters
    ----------
    experiment: string
        The name of the model

    Returns
    ----------
    epoch: int
        The last epoch
    """
    files = os.listdir(f'../results_{experiment}')
    epoch = 0
    for f in files:
        if '_' in f:
            fepoch = int(f.split('_')[0])
            if fepoch > epoch:
                epoch = fepoch
    return epoch

if __name__ == "__main__":
    pass
