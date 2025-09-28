# 1) clean the old links
rm -f output/kaist_input/*.nii.gz
mkdir -p output/kaist_input

# 2) recreate ABSOLUTE symlinks and use _000X naming (duplicate T1CE into all 4 slots)
for id in 71899681 76686586 77364735 77402268 79391843; do
  src=$(realpath "output/nifti/$id/${id}_T1_CE_3D_AX_ALIGNED.nii.gz")
  ln -s "$src" "output/kaist_input/${id}_0000.nii.gz"  # flair  (placeholder)
  ln -s "$src" "output/kaist_input/${id}_0001.nii.gz"  # t1     (placeholder)
  ln -s "$src" "output/kaist_input/${id}_0002.nii.gz"  # t1ce   (real modality)
  ln -s "$src" "output/kaist_input/${id}_0003.nii.gz"  # t2     (placeholder)
done

# 3) sanity check: do we have 20 files and all links resolve?
ls -l output/kaist_input | head
find output/kaist_input -type l -exec test -e {} \; -print | wc -l   # should print 20
