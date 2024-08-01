
import streamlit as st
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import tifffile as tiff
import torch.nn as nn
import numpy as np

# Define a function to load your model
class seg_ConvDeconvNet(nn.Module):
    def __init__(self, model):
        super(seg_ConvDeconvNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1),  # Output: (16, H, W)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (16, H/2, W/2)
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # Output: (32, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (32, H/4, W/4)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),  # Output: (16, H/2, W/2)
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=2, stride=2),  # Output: (3, H, W)
            nn.ReLU()
        )
        
        self.segmentation_model = model  # Assign the model parameter

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.segmentation_model(x)
        return x

def load_model(model_path):
    model = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', encoder_output_stride=16, decoder_channels=128)
    model = seg_ConvDeconvNet(model)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'),weights_only=True))
    model.eval()
    return model

# Define a function to perform inference
def predict_image(model, image):
    image = image.unsqueeze(0).permute(0, 3, 1, 2)  # Add batch dimension and permute dimensions
  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        output = output.squeeze(0)  # Remove batch dimension
        output = output.cpu().numpy()
        output = (output > 0.5).astype(np.uint8)
    return output



def convert_to_rgb(mask):
    """
    Convert a single-channel mask to a 3-channel RGB image for visualization.
    
    Parameters:
    - mask (numpy.ndarray): Input mask.
    
    Returns:
    - numpy.ndarray: RGB image.
    """
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[mask == 1] = [255, 255, 0]  # Yellow color for mask
    return rgb_mask

def normalize(image):
    """
    Normalize the image to the range [0, 1].
    
    Parameters:
    - image (numpy.ndarray): Input image.
    
    Returns:
    - numpy.ndarray: Normalized image.
    """
    # Normalize each band to [0, 1]
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

# Streamlit app layout
st.title("Satellite Image Segmentation")

uploaded_file = st.file_uploader("Choose a TIFF image...")
if uploaded_file is not None:
    image = tiff.imread(uploaded_file)
    image_array = np.array(image)
    image_array_normal = normalize(image_array)
    for i in range(3):
        band = image_array_normal[:, :, i]
        band = (band * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
        st.image(band, caption=f'Band {i+1}', use_column_width=True)

    image_array_tensor = torch.tensor(image_array_normal).float()
    # Load the model
    model = load_model('DeepLabV3Plus_encoder_decoder_model.pth')
    # Perform prediction
    segmented_image = predict_image(model, image_array_tensor)
    # Display the segmented image
    segmented_image = segmented_image.squeeze(0)
    segmented_image_rgb = convert_to_rgb(segmented_image)  # Convert to RGB
    
    # Display the segmented image
    st.image(segmented_image_rgb, caption='Segmented Image', use_column_width=True)
