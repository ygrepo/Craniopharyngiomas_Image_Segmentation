# --- IMPORTS ---
import torch
import numpy as np
from pathlib import Path


from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.imageio.image_reader_writer import load_volume

# --- CONFIGURATION ---
model_folder = (
    "nnUNet_results/Dataset503_CP/"
    "nnUNetTrainerEarlyStopping__nnUNetResEncUNetMPlans__3d_fullres"
)
input_image_path = "nnUNet_raw/Dataset503_CP/imagesTs/06780898_0000.nii.gz"
case_id = Path(input_image_path).stem.rsplit("_", 1)[0]  # '06780898'

images_dir = Path("nnUNet_raw/Dataset503_CP/imagesTs")

# Channel order must match dataset.json:
# 0: FLAIR, 1: T1CE, 2: T2
channel_paths = [
    images_dir / f"{case_id}_0000.nii.gz",  # FLAIR
    images_dir / f"{case_id}_0001.nii.gz",  # T1CE
    images_dir / f"{case_id}_0002.nii.gz",  # T2
]

output_dir = (
    "nnUNet_results/Dataset503_CP/"
    "nnUNetTrainerEarlyStopping__nnUNetResEncUNetMPlans__3d_fullres/"
    "fold_0/latent_features"
)
checkpoint_name = "checkpoint_final.pth"
device = torch.device("cuda")

print(f"Using device: {device}")
print(f"Model folder: {model_folder}")
print("Using channels:")
for p in channel_paths:
    print("  ", p)

# --- STEP 1: Load predictor / model ---
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=False,
    perform_everything_on_device=True,  # nnUNetv2 arg name
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

# Pass all 3 channels in correct order
images, props = io.read_images([str(p) for p in channel_paths])

# Instantiate the preprocessor using the configuration from the loaded model
# This ensures you use the exact normalization/resampling the model expects
preprocessor_class = predictor.configuration_manager.preprocessor_class
preprocessor = preprocessor_class(
    verbose=predictor.verbose,
    configuration_manager=predictor.configuration_manager,
    label_manager=predictor.plans_manager.get_label_manager(predictor.dataset_json),
)

# Run preprocessing
# data is (C, Z, Y, X), seg is None (since we are testing)
data, seg, props = preprocessor.preprocess_test_case(images, props)

# nnU-Net preprocessor returns numpy array. Convert to tensor.
img_np = data.astype(np.float32)
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
    f"Feature vector stats - min: {lasso_input_vector.min():.4f}, "
    f"max: {lasso_input_vector.max():.4f}, "
    f"mean: {lasso_input_vector.mean():.4f}"
)

# Save the feature vector
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{case_id}.npy"
np.save(output_path, lasso_input_vector)
print(f"Saved feature vector to: {output_path}")
