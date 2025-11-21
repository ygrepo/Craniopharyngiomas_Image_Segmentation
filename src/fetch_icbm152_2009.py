from nilearn import datasets

mni = datasets.fetch_icbm152_2009(
    data_dir="mni152", verbose=1  # or any folder you want
)

print(mni.keys())
print("MNI T1 path:", mni["t1"])
