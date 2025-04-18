# MRI-CT Registration using CNN

This project implements a CNN-based approach for registering MRI and CT medical images. The registration is performed using a deep learning model that learns to align the images by predicting a displacement field.

## Project Structure

```
MRI_CT_registration/
├── models/
│   ├── __init__.py
│   └── registration_model.py
├── utils/
│   ├── __init__.py
│   └── data_processing.py
├── train.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8 or higher
- PyTorch
- Nibabel
- Scikit-image
- Matplotlib

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your MRI and CT images in NIfTI format (.nii or .nii.gz)

2. Run the training script:
```bash
python train.py
```

3. The script will:
   - Load and preprocess the images
   - Train the registration model
   - Apply the learned transformation
   - Visualize the results

## Model Architecture

The registration model consists of:
- An encoder network that processes both fixed and moving images
- A decoder network that predicts the displacement field
- The model uses 3D convolutions to handle volumetric medical images

## Data Format

The code expects input images in NIfTI format (.nii or .nii.gz). The images should be preprocessed to have similar dimensions and intensity ranges.

## Notes

- The current implementation uses a simple CNN architecture. For better results, you might want to:
  - Use a more sophisticated architecture (e.g., VoxelMorph)
  - Implement multi-scale registration
  - Add regularization to the displacement field
  - Use a more sophisticated loss function