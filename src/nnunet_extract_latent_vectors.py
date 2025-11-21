# --- IMPORTS ---
import torch
import numpy as np
from pathlib import Path

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

# --- CONFIGURATION ---
model_folder = "nnUNet_results/Dataset503_CP/nnUNetTrainerEarlyStopping__nnUNetResEncUNetMPlans__3d_fullres"
input_image_path = "nnUNet_raw/Dataset503_CP/imagesTs/06975111_0000.nii.gz"
output_dir = "nnUNet_results/Dataset503_CP/nnUNetTrainerEarlyStopping__nnUNetResEncUNetMPlans__3d_fullres/fold_0/latent_features"
checkpoint_name = "checkpoint_final.pth"
device = torch.device("cuda")


print(f"Using device: {device}")
print(f"Model folder: {model_folder}")
print(f"Input image: {input_image_path}")


# --- STEP 1: Load predictor / model ---
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=False,
    perform_everything_on_device=True,
    device=device,
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=False,
)

predictor.initialize_from_trained_model_folder(
    model_folder,
    use_folds=(0,),  # or tuple of folds
    checkpoint_name=checkpoint_name,
)

network = predictor.network
network.eval()
network.to(device)

# --- STEP 2: Register hook on bottleneck ---
latent_features = {}


def get_activation(name):
    def hook(module, inp, out):
        latent_features[name] = out.detach().cpu()

    return hook


bottleneck_module = network.encoder.stages[-1]
handle = bottleneck_module.register_forward_hook(get_activation("bottleneck"))
print(f"Hook registered on: {bottleneck_module}")

# --- STEP 3: Read and preprocess the case ---
io = SimpleITKIO()
# nnU-Net expects list of filepaths (per channel). If single-channel:
images, props = io.read_images([input_image_path])

# Use the predictor's preprocessor; actual method name may differ slightly
# Check predictor.preprocessor for available methods.
preprocessed = predictor.preprocessor.preprocess_single_case(images, props)
# preprocessed[0] is (C, Z, Y, X)
img_np = preprocessed[0].astype(np.float32)
input_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)  # (1, C, D, H, W)

# --- STEP 4: Forward pass to trigger hook ---

with torch.no_grad():
    try:
        _ = network(input_tensor)
    except Exception as e:
        print(f"Forward pass failed: {e}")
        handle.remove()
        raise

# Check if hook was triggered
if "bottleneck" not in latent_features:
    handle.remove()
    raise RuntimeError("Hook was not triggered - check bottleneck module selection")

raw_features = latent_features["bottleneck"]  # (1, C, D, H, W)
print("Raw bottleneck shape:", raw_features.shape)

# Global average pooling over spatial dims
pooled_features = torch.mean(raw_features, dim=(2, 3, 4)).squeeze()  # (C,)
print("Final feature vector shape:", pooled_features.shape)

# Clean up hook
handle.remove()

# Convert to numpy for LASSO feature vector
lasso_input_vector = pooled_features.numpy()
print("Feature vector extracted successfully")
print(
    f"Feature vector stats - min: {lasso_input_vector.min():.4f}, max: {lasso_input_vector.max():.4f}, mean: {lasso_input_vector.mean():.4f}"
)

# Save the feature vector
output_path = f"{output_dir}/{Path(input_image_path).stem}.npy"
np.save(output_path, lasso_input_vector)
print(f"Saved feature vector to: {output_path}")
