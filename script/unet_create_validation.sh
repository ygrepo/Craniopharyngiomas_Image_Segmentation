#!/bin/bash
PREP=/projects/gbm_modeling/github/Craniopharyngiomas_Image_Segmentation/nnUNet_preprocessed/Dataset501_BraTS2017_4ch

python - <<'PY'
import json, os
splits = json.load(open(os.path.join(os.environ["PREP"], "splits_final.json")))
val_ids = set(splits[0]["val"])
open("val_ids_fold0.txt","w").write("\n".join(sorted(val_ids)))
print(f"Wrote {len(val_ids)} IDs to val_ids_fold0.txt")
PY
