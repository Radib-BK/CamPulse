import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Create alignment object
align = rs.align(rs.stream.color)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert depth image to 3-channel image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack color and depth images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.imshow('RealSense', images)

        # Get cursor position and depth
        def onMouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Print pixel coordinates
                print("Pixel Coordinates (x, y):", x, y)
                # Print depth value
                depth = depth_frame.get_distance(x, y)
                print("Distance from camera:", depth, "meters")

        cv2.setMouseCallback('RealSense', onMouse)

        # Close on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
