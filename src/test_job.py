import nibabel as nib

img = nib.load("output/preprocessed/71899681/71899681_T1_CE_3D_AX_ALIGNED.nii.gz")
print("Shape:", img.shape)
print("Voxel spacing:", img.header.get_zooms())
print("Orientation:", nib.aff2axcodes(img.affine))
