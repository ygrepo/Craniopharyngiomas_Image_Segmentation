import pandas as pd
import numpy as np
from pathlib import Path
import os

# --- CONFIGURATION ---
# Paths to your data
latent_dir = Path("nnUNet_results/Dataset503_CP/.../latent_features")
radiomics_csv = Path("nnUNet_raw/Dataset503_CP/radiomics_results.csv")
clinical_csv = Path("clinical_data.csv")  # Make sure you have this!

# --- PART 1: Load Latent Features into a DataFrame ---
print("Loading latent features...")
latent_data = []

# Iterate through all .npy files
for npy_file in sorted(latent_dir.glob("*.npy")):
    case_id = npy_file.stem  # e.g., "06780898"

    # Load the (320,) vector
    vector = np.load(npy_file)

    # Create a dictionary for this row
    row = {"Case_ID": case_id}
    # Add columns Latent_0, Latent_1, ..., Latent_319
    for i, val in enumerate(vector):
        row[f"Latent_{i}"] = val

    latent_data.append(row)

df_latent = pd.DataFrame(latent_data)
print(f"Latent DataFrame shape: {df_latent.shape}")
# Expect: (N_patients, 321) -> 320 features + Case_ID

# --- PART 2: Load Radiomics ---
print("Loading radiomics...")
df_rad = pd.read_csv(radiomics_csv)
# Ensure Case_ID is string to match latent data
df_rad["Case_ID"] = df_rad["Case_ID"].astype(str)

# --- PART 3: Load Clinical & Outcome (The Target) ---
# You need this for LASSO to learn anything!
# Assuming columns: Case_ID, Age, Sex, Visual_Outcome
if clinical_csv.exists():
    df_clin = pd.read_csv(clinical_csv)
    df_clin["Case_ID"] = df_clin["Case_ID"].astype(str)
else:
    print("WARNING: Clinical data missing. Creating dummy target for demo.")
    df_clin = pd.DataFrame(
        {
            "Case_ID": df_latent["Case_ID"],
            "Visual_Outcome": np.random.randint(
                0, 2, size=len(df_latent)
            ),  # 0=Unchanged, 1=Worsened
        }
    )

# --- PART 4: MERGE EVERYTHING ---
# Merge Latent + Radiomics
df_master = pd.merge(df_rad, df_latent, on="Case_ID", how="inner")

# Merge + Clinical
df_master = pd.merge(df_master, df_clin, on="Case_ID", how="inner")

print(f"Final Master Table Shape: {df_master.shape}")
print(df_master.head())

# Save for safety
df_master.to_csv("MASTER_LASSO_INPUT.csv", index=False)
