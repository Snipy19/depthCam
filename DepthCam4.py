import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# Configuration
IMAGE_SIZE = (640, 480)
QUALITY = 75
DISPARITY_RANGE = 16 * 5  
BLOCK_SIZE = 3      
COLORMAP = cv2.COLORMAP_TURBO
OUTPUT_FILE = 'disparity_map.jpg'

def resize_and_save(image_path, output_size, quality):
    """Thread function to resize and save a single image"""
    img = Image.open(image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    img_resized = img.resize(output_size, Image.Resampling.LANCZOS)
    out_path = f"resized_{image_path.split('/')[-1].split('.')[0]}.jpg"
    img_resized.save(out_path, optimize=True, quality=quality)
    return out_path

def resize_and_compress_two_images(image1_path, image2_path):
    """Parallel image resizing"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(resize_and_save, image1_path, IMAGE_SIZE, QUALITY)
        future2 = executor.submit(resize_and_save, image2_path, IMAGE_SIZE, QUALITY)
        return future1.result(), future2.result()

def compute_disparity(grayL, grayR):
    """Compute disparity map optimized for Raspberry Pi"""
    stereo = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=DISPARITY_RANGE,
        blockSize=BLOCK_SIZE,
        P1=8 * 1 * BLOCK_SIZE * BLOCK_SIZE,
        P2=32 * 1 * BLOCK_SIZE * BLOCK_SIZE,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return stereo.compute(grayL, grayR).astype(np.float32) / 16.0

def main():
    # Load and resize images (parallel)
    res_left, res_right = resize_and_compress_two_images('lekt.jpg', 'rikt.jpg')
    
    # Read grayscale images
    grayL = cv2.cvtColor(cv2.imread(res_left), cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(cv2.imread(res_right), cv2.COLOR_BGR2GRAY)
    
    # Match dimensions if needed
    if grayL.shape != grayR.shape:
        grayR = cv2.resize(grayR, (grayL.shape[1], grayL.shape[0]))

    # Compute disparity
    start_time = time.time()
    disp = compute_disparity(grayL, grayR)
    elapsed = time.time() - start_time
    print(f"Disparity computation time: {1/elapsed:.4f} FPS")
    print(f"Disparity range: {np.min(disp)} to {np.max(disp)}")

    # Post-process and save
    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_inv = 255 - disp_norm  # Invert for better visualization
    disp_color = cv2.applyColorMap(disp_inv, COLORMAP)
    cv2.imwrite(OUTPUT_FILE, disp_color)
    print(f"Saved disparity map to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
