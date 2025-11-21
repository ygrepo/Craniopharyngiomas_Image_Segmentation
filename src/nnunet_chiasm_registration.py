import ants
from pathlib import Path

# --- Paths ---
base = Path("nnUNet_raw/Dataset503_CP")
imagesTr = base / "imagesTr"
imagesTs = base / "imagesTs"

# IMPORTANT: use the 09a atlas and your manual chiasm mask
mni_t1_path = Path("./data/CP/atlas/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz")
mni_chiasm_path = Path("./data/CP/atlas/chiasm.nii")

outdir = base / "chiasm_masks"
outdir.mkdir(exist_ok=True)

# Load atlas once
mni_t1 = ants.image_read(str(mni_t1_path)).reorient_image2("RAI")
chiasm_mni = ants.image_read(str(mni_chiasm_path)).reorient_image2("RAI")


def process_split(split_dir):
    # all T1CE files: channel 1
    t1_files = sorted(split_dir.glob("*_0001.nii.gz"))

    for t1_path in t1_files:
        case_id = t1_path.name.replace("_0001.nii.gz", "")
        out_file = outdir / f"{case_id}_chiasm_mask.nii.gz"

        if out_file.exists():
            print(f"[SKIP] {case_id}")
            continue

        print(f"[RUN]  {case_id}")
        patient_t1 = ants.image_read(str(t1_path)).reorient_image2("RAI")

        # --- Registration: atlas -> patient ---
        reg = ants.registration(
            fixed=patient_t1,
            moving=mni_t1,
            type_of_transform="SyN",
            verbose=False,
        )

        # --- Warp chiasm mask into patient space ---
        chiasm_patient = ants.apply_transforms(
            fixed=patient_t1,
            moving=chiasm_mni,
            transformlist=reg["fwdtransforms"],
            interpolator="nearestNeighbor",
        )

        ants.image_write(chiasm_patient, str(out_file))
        print(f"  → saved {out_file}")


# Run
process_split(imagesTr)
process_split(imagesTs)
