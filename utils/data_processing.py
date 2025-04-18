import nibabel as nib
import numpy as np
from skimage.transform import resize
from scipy.ndimage import affine_transform
import torch

def load_nifti(file_path):
    """Load NIfTI file and return the image data."""
    img = nib.load(file_path)
    return img.get_fdata(), img.affine

def preprocess_image(image, target_shape=(128, 128, 128)):
    """Preprocess image by resizing and normalizing."""
    # Resize image
    resized = resize(image, target_shape, anti_aliasing=True)
    # Normalize to [0, 1]
    normalized = (resized - resized.min()) / (resized.max() - resized.min())
    return normalized

def apply_displacement_field(image, displacement_field):
    """Apply displacement field to image."""
    # Convert displacement field to numpy
    disp = displacement_field.squeeze().detach().cpu().numpy()
    
    # Create grid of coordinates
    grid = np.meshgrid(np.arange(image.shape[2]), 
                       np.arange(image.shape[3]), 
                       np.arange(image.shape[4]), 
                       indexing='ij')
    
    # Apply displacement
    warped_coords = [grid[i] + disp[i] for i in range(3)]
    
    # Interpolate
    warped_image = affine_transform(image.squeeze().cpu().numpy(), 
                                  warped_coords, 
                                  order=1)
    
    return torch.FloatTensor(warped_image).unsqueeze(0).unsqueeze(0) 