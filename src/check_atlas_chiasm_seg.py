import nibabel as nib
import numpy as np
import sys


def load_info(path):
    img = nib.load(path)
    hdr = img.header
    return {
        "shape": hdr.get_data_shape(),
        "zooms": hdr.get_zooms(),
        "affine": img.affine,
    }


def compare(a, b, name_a="ATLAS", name_b="MASK"):
    print(f"\n=== Comparing {name_a} and {name_b} ===")

    print("\nDimensions:")
    print(f"{name_a}: {a['shape']}")
    print(f"{name_b}: {b['shape']}")
    print("Match:", a["shape"] == b["shape"])

    print("\nVoxel spacing:")
    print(f"{name_a}: {a['zooms']}")
    print(f"{name_b}: {b['zooms']}")
    print("Match:", np.allclose(a["zooms"], b["zooms"]))

    print("\nAffine:")
    print(f"{name_a} affine:\n{a['affine']}\n")
    print(f"{name_b} affine:\n{b['affine']}\n")
    print("Affine match:", np.allclose(a["affine"], b["affine"]))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_geometry.py <atlas.nii.gz> <mask.nii.gz>")
        sys.exit(1)

    atlas_file = sys.argv[1]
    mask_file = sys.argv[2]

    print("Loading files...")
    atlas_info = load_info(atlas_file)
    mask_info = load_info(mask_file)

    compare(atlas_info, mask_info)
