"""
Image visualization utilities for brain tumor segmentation.

This module provides functions for visualizing medical images,
segmentation masks, and model predictions.
"""
from nilearn import plotting, image
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from typing import Optional, Tuple, List

import random


def display_multiple_slices(
                          image_file: str,
                          label_file: str,
                          slice_indices: List[int] = None,
                          title: str = "Brain MRI Slices",
                          cmap: str = "gray",
                          figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Display multiple slices from 3D image data.
    
    Args:
        image_str: path to image file
        label_file: path to label file
        slice_indices: List of slice indices to display
        title: Title for the plot
        cmap: Colormap for visualization
        figsize: Figure size (width, height)
    """

    # Multiple views at once
    img_4d = nib.load(image_file)
    seg_3d = nib.load(label_file)
    img_timepoint0 = image.index_img(img_4d, 0)

    plotting.plot_epi(img_timepoint0, display_mode='ortho', title='Brain Scan')
    plt.show()

    plotting.plot_roi(seg_3d, bg_img=img_timepoint0, title='Tumor Mask')
    plt.show()


def load_and_visualize_nifti(file_path: str,
                            label_file: str) -> np.ndarray:
    """
    Load NIfTI file and visualize a slice.
    
    Args:
        file_path: Path to NIfTI file
        slice_idx: Index of slice to display
        title: Title for visualization (defaults to filename)
        
    Returns:
        Loaded image data as numpy array
    """
    # Load NIfTI file
    img = nib.load(file_path)
    image_data = img.get_fdata()
    print(f"image data shape: {image_data.shape}")
    
    # Display the slice
    #display_three_views_with_channels(image_data, slice_idx, title)
    display_multiple_slices(file_path, label_file)
    
    return image_data


if __name__ == "__main__":
    # Example usage
    random_number = random.randint(0, 9)
    # Load and visualize NIfTI file
    file_path = f"imagesTr/BRATS_00{random_number}.nii.gz"
    label_file = f"labelsTr/BRATS_00{random_number}.nii.gz"
    load_and_visualize_nifti(file_path, label_file)