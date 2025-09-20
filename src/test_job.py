import nibabel as nib, numpy as np

m = nib.load("output/mri_core_3d/71899681/pred_mask.nii.gz")
a = m.get_fdata()
print("unique:", np.unique(a), "voxels>0:", int((a > 0.5).sum()))
print("shape:", a.shape)
print(
    "qform set:",
    m.get_qform()[0] is not None,
    "sform set:",
    m.get_sform()[0] is not None,
)
print("axcodes:", nib.aff2axcodes(m.affine))
