import cv2
import numpy as np
import time

# === Stereo Camera Parameters ===
focal_length_mm = 4.0      # Focal length in mm
baseline_m = 0.1           # Baseline in meters
sensor_width_mm = 3.68     # Typical for 1/3" sensor (adjust if needed)
image_width_px = 640       # Adjust based on your resolution

# Convert focal length from mm to pixels
focal_length_px = (focal_length_mm / sensor_width_mm) * image_width_px

# === Open two camera streams ===
capL = cv2.VideoCapture(0)  # Left camera
capR = cv2.VideoCapture(1)  # Right camera

# Set resolution
capL.set(cv2.CAP_PROP_FRAME_WIDTH, image_width_px)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, image_width_px)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# === Stereo Matcher ===
min_disp = 0
num_disp = 16 * 4
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

# === FPS Calculation ===
prev_time = time.time()
fps = 0

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("Error: Unable to read from both cameras")
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Compute disparity
    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Normalize disparity for visualization
    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)

    # === Depth Calculation ===
    with np.errstate(divide='ignore'):  # Avoid division by zero
        depth_map = (focal_length_px * baseline_m) / (disp + 1e-6)  # Depth in meters

    # Calculate FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # Overlay FPS
    cv2.putText(disp_color, f"FPS: {fps:.4f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Overlay depth info at center pixel
    h, w = depth_map.shape
    center_depth = depth_map[h//2, w//2]
    cv2.putText(disp_color, f"Depth: {center_depth:.2f} m", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show results
    cv2.imshow("Left", frameL)
    cv2.imshow("Right", frameR)
    cv2.imshow("Disparity Map", disp_color)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
