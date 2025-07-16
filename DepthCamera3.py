import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

def resize_and_compress_two_images(image1_path, image2_path, output_size, quality=75):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    if img1.mode == 'RGBA': img1 = img1.convert('RGB')
    if img2.mode == 'RGBA': img2 = img2.convert('RGB')
    img1_resized = img1.resize(output_size, Image.Resampling.LANCZOS)
    img2_resized = img2.resize(output_size, Image.Resampling.LANCZOS)
    out1 = f"resized_{image1_path.split('/')[-1].split('.')[0]}.jpg"
    out2 = f"resized_{image2_path.split('/')[-1].split('.')[0]}.jpg"
    img1_resized.save(out1, optimize=True, quality=quality)
    img2_resized.save(out2, optimize=True, quality=quality)
    print(f"Saved: {out1}, {out2}")
    return out1, out2

# === Load stereo images ===
res_left, res_right = resize_and_compress_two_images('lekt.jpg', 'rikt.jpg', (640,480), quality=75)
imgL = cv2.imread(res_left)
imgR = cv2.imread(res_right)
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
if grayL.shape != grayR.shape:
    grayR = cv2.resize(grayR, (grayL.shape[1], grayL.shape[0]))

# === Disparity computation ===
start_time = time.time()
min_disp = 0
num_disp = 16 * 5
block = 5
stereo = cv2.StereoSGBM.create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block,
    P1=8 * 1 * block * block,
    P2=32 * 1 * block * block,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
elapsed = time.time() - start_time
print(f"Disparity compute time: {1/(elapsed):.4f} sec")

# === Normalize and invert disparity ===
disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
disp_inv = 255 - disp_norm  # invert mapping so high disparity becomes warm colors

# Apply vibrant colormap
disp_color = cv2.applyColorMap(disp_inv, cv2.COLORMAP_TURBO)
disp_color_rgb = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)

# === Plot with colorbar of disparity values ===
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(disp_color_rgb)
ax.set_title('Colored Disparity Map (Warm=Close, Cool=Far)')
ax.axis('off')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Normalized Disparity')
cbar.locator = MaxNLocator(nbins=10)
cbar.update_ticks()

plt.tight_layout()
plt.show()

print("Disparity min:", np.min(disp), "max:", np.max(disp))
