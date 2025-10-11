import json, SimpleITK as sitk, numpy as np

p = "summary.json"
with open(p) as f:
    data = json.load(f)
items = (
    data if isinstance(data, list) else (data.get("results") or data.get("cases") or [])
)


def uniq(img):
    a = sitk.GetArrayFromImage(img)
    vals, cnts = np.unique(a, return_counts=True)
    return dict(zip(vals.tolist(), cnts.tolist()))


for it in items[:5]:  # sample a few cases
    pred = sitk.ReadImage(it["prediction_file"])
    ref = sitk.ReadImage(it["reference_file"])
    print("CASE:", it.get("case_id", it["prediction_file"]))
    print("  pred unique:", uniq(pred))  # should include 1,2,3 sometimes (not only 0)
    print("  ref  unique:", uniq(ref))  # should include 1,2,3 for that case if present
    for cid in (1, 2, 3):
        pb = sitk.Equal(pred, cid)
        rb = sitk.Equal(ref, cid)
        sf = sitk.StatisticsImageFilter()
        sf.Execute(pb)
        p_any = sf.GetSum() > 0
        sf.Execute(rb)
        r_any = sf.GetSum() > 0
        print(f"   class {cid}: pred_any={p_any}, ref_any={r_any}")
