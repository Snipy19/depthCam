import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Configuration
IMAGE_SIZE = (640, 480)
DISPARITY_RANGE = 16 * 5
BLOCK_SIZE = 3
COLORMAP = cv2.COLORMAP_TURBO

def resize_frame(frame, size):
    """Resize OpenCV frame using PIL for better quality."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_resized = img.resize(size, Image.Resampling.LANCZOS)
    return cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

def compute_disparity(grayL, grayR):
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
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    return disparity

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def main():
    # Open cameras
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(2)

    # Set resolution
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Could not open both cameras.")
        return

    executor = ThreadPoolExecutor(max_workers=2)

    print("Press 'q' to quit.")

    while True:
        # Grab frames in parallel
        future_left = executor.submit(capture_frame, cap_left)
        future_right = executor.submit(capture_frame, cap_right)

        frame_left = future_left.result()
        frame_right = future_right.result()

        if frame_left is None or frame_right is None:
            print("Failed to grab frames.")
            break

        # Resize frames (optional since we set camera resolution, but to ensure consistent size)
        frame_left = resize_frame(frame_left, IMAGE_SIZE)
        frame_right = resize_frame(frame_right, IMAGE_SIZE)

        # Convert to grayscale
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        disparity = compute_disparity(gray_left, gray_right)

        # Normalize and color map
        disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_color = cv2.applyColorMap(disp_norm.astype(np.uint8), COLORMAP)

        # Show disparity map
        cv2.imshow("Disparity Map", disp_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()