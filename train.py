import torch
import torch.optim as optim
from models.registration_model import RegistrationCNN
from utils.data_processing import load_nifti, preprocess_image, apply_displacement_field
import matplotlib.pyplot as plt

def train_model(model, fixed_image, moving_image, num_epochs=100, learning_rate=0.001):
    """Train the registration model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    fixed_tensor = torch.FloatTensor(fixed_image).unsqueeze(0).unsqueeze(0).to(device)
    moving_tensor = torch.FloatTensor(moving_image).unsqueeze(0).unsqueeze(0).to(device)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        displacement_field = model(fixed_tensor, moving_tensor)
        
        # Apply displacement field to moving image
        warped_image = apply_displacement_field(moving_tensor, displacement_field)
        
        # Calculate loss
        loss = criterion(warped_image, fixed_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model, displacement_field

def plot_images(fixed, moving, warped, slice_idx=64):
    """Plot fixed, moving, and warped images."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(fixed[slice_idx, :, :], cmap='gray')
    axes[0].set_title('Fixed Image')
    
    axes[1].imshow(moving[slice_idx, :, :], cmap='gray')
    axes[1].set_title('Moving Image')
    
    axes[2].imshow(warped[slice_idx, :, :], cmap='gray')
    axes[2].set_title('Warped Image')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Example usage
    mri_path = 'path_to_mri.nii.gz'
    ct_path = 'path_to_ct.nii.gz'
    
    # Load and preprocess images
    mri_image, _ = load_nifti(mri_path)
    ct_image, _ = load_nifti(ct_path)
    
    mri_processed = preprocess_image(mri_image)
    ct_processed = preprocess_image(ct_image)
    
    # Initialize model
    model = RegistrationCNN()
    
    # Train model
    trained_model, displacement_field = train_model(model, mri_processed, ct_processed)
    
    # Apply transformation
    warped_image = apply_displacement_field(torch.FloatTensor(ct_processed).unsqueeze(0).unsqueeze(0), displacement_field)
    
    # Visualize results
    plot_images(mri_processed, ct_processed, warped_image.squeeze().numpy()) 