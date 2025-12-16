# Retry evaluation with fixes:
# - Ensure images have minimum size for SSIM (resize if needed).
# - Use channel_axis=2 and data_range=255 in ssim.
# - Recreate outputs.

from pathlib import Path
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import math
import pandas as pd
import matplotlib.pyplot as plt
import os

out_dir = Path("/mnt/data/eval_outputs")
out_dir.mkdir(parents=True, exist_ok=True)

# rewrite evaluate.py (improved) to disk
evaluate_py = r'''
# (Improved evaluate.py - robust SSIM handling and utilities)
import argparse
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math
import pandas as pd
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def ensure_min_size(img, min_side=64):
    h,w = img.shape[:2]
    if min(h,w) < min_side:
        scale = min_side / min(h,w)
        neww = int(round(w*scale))
        newh = int(round(h*scale))
        img = cv2.resize(img, (neww,newh), interpolation=cv2.INTER_LINEAR)
    return img

def psnr(img1, img2, data_range=255.0):
    mse = ((img1.astype('float64') - img2.astype('float64')) ** 2).mean()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(data_range / math.sqrt(mse))

def ssim_metric(img1, img2):
    img1 = ensure_min_size(img1.copy())
    img2 = ensure_min_size(img2.copy())
    return float(ssim(img1, img2, channel_axis=2, data_range=255))

def contrast_score(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:,:,0].astype('float32')
    return float(L.std())
'''
(Path("/mnt/data/evaluate.py")).write_text(evaluate_py)

# Load provided image
img_path = Path("/mnt/data/60c90190-e2be-4d98-907b-b0c5558d599c.png")
orig = cv2.imread(str(img_path))
if orig is None:
    raise FileNotFoundError("Provided image not found.")
orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

# Create demo processed outputs
hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
v_clahe = clahe.apply(v)
hsv_clahe = cv2.merge([h,s,v_clahe])
classical_bgr = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
classical_rgb = cv2.cvtColor(classical_bgr, cv2.COLOR_BGR2RGB)

bfilter = cv2.bilateralFilter(orig, d=9, sigmaColor=75, sigmaSpace=75)
b_rgb = cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB)
gamma = 1.05
table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
deep_rgb = cv2.LUT(b_rgb, table)

# Save demo outputs
demo_folder = out_dir / "demo_preds"
demo_folder.mkdir(exist_ok=True)
orig_save = out_dir / "orig.png"
classical_save = demo_folder / "classical.png"
deep_save = demo_folder / "deep.png"
cv2.imwrite(str(orig_save), cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(classical_save), cv2.cvtColor(classical_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(deep_save), cv2.cvtColor(deep_rgb, cv2.COLOR_RGB2BGR))

# Metrics functions (robust)
from skimage.metrics import structural_similarity as ssim_func
def ensure_min_size(img, min_side=64):
    h,w = img.shape[:2]
    if min(h,w) < min_side:
        scale = min_side / min(h,w)
        neww = int(round(w*scale))
        newh = int(round(h*scale))
        img = cv2.resize(img, (neww,newh), interpolation=cv2.INTER_LINEAR)
    return img

def psnr_func(img1, img2, data_range=255.0):
    img1 = ensure_min_size(img1)
    img2 = ensure_min_size(img2)
    mse = ((img1.astype('float64') - img2.astype('float64')) ** 2).mean()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(data_range / math.sqrt(mse))

def ssim_metric_local(img1, img2):
    img1 = ensure_min_size(img1)
    img2 = ensure_min_size(img2)
    return float(ssim_func(img1, img2, channel_axis=2, data_range=255))

def contrast_score_local(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:,:,0].astype('float32')
    return float(L.std())

def vqs_local(ref, pred):
    s = ssim_metric_local(ref, pred)
    c_ref = contrast_score_local(ref)
    c_pred = contrast_score_local(pred)
    if c_ref == 0:
        cr = 0.0
    else:
        cr = (c_pred - c_ref) / c_ref
    cr = max(min(cr,1.0), -1.0)
    return float(0.7 * s + 0.3 * (0.5*(cr+1.0)))

records = []
for name, img in [("classical", classical_rgb), ("deep", deep_rgb)]:
    p = psnr_func(orig_rgb, img)
    s = ssim_metric_local(orig_rgb, img)
    v = vqs_local(orig_rgb, img)
    c_ref = contrast_score_local(orig_rgb)
    c_pred = contrast_score_local(img)
    records.append({"image": img_path.name, "method": name, "PSNR": p, "SSIM": s, "VQS": v, "Contrast_ref": c_ref, "Contrast_pred": c_pred})

df = pd.DataFrame.from_records(records)
csv_path = out_dir / "evaluation_results.csv"
df.to_csv(csv_path, index=False)

# Plot
agg = df.groupby("method").mean()[["PSNR","SSIM","VQS"]]
fig, ax = plt.subplots(figsize=(7,4))
agg.plot.bar(rot=0, ax=ax)
ax.set_title("Average Metrics per Method (Demo)")
plt.tight_layout()
plot_path = out_dir / "metrics_plot.png"
plt.savefig(plot_path)

from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("Evaluation Results (Demo)", df)

generated_files = {
    "evaluate_py": "/mnt/data/evaluate.py",
    "csv": str(csv_path),
    "plot": str(plot_path),
    "demo_orig": str(orig_save),
    "demo_classical": str(classical_save),
    "demo_deep": str(deep_save)
}
generated_files

