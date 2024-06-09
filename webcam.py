import cv2
import numpy as np
import time
import pyrealsense2 as rs

class Webcam(object):
    def __init__(self):
        #print ("WebCamEngine init")
        self.dirname = "" #for nothing, just to make 2 inputs the same
        self.cap = None
        self.depth_pipeline = None

    
    
    def start(self):
        print("[INFO] Start webcam")
        time.sleep(1) # wait for camera to be ready
        self.cap = cv2.VideoCapture(2)
        self.valid = False
        try:
            resp = self.cap.read()
            self.shape = resp[1].shape
            self.valid = True
        except:
            self.shape = None
            
            
            
        # Initialize the RealSense pipeline for depth
        self.depth_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.depth_pipeline.start(config)
    
    def get_frame(self):
    
        if self.valid:
            _,frame = self.cap.read()
            frame = cv2.flip(frame,1)
            # cv2.putText(frame, str(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            #           (65,220), cv2.FONT_HERSHEY_PLAIN, 2, (0,256,256))
        else:
            frame = np.ones((480,640,3), dtype=np.uint8)
            col = (0,256,256)
            cv2.putText(frame, "(Error: Camera not accessible)",
                       (65,220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame
    
    def depth_frame_distance(self):
        if self.depth_pipeline:
            # Wait for a coherent pair of frames: depth and color
            frames = self.depth_pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            
            # Convert depth frame to numpy array
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Get the height and width of the depth image
                height, width = depth_image.shape
                
                # Calculate the middle point indices
                middle_row = height // 2
                middle_col = width // 2
                
                # Get the depth value at the middle point
                depth_value = depth_image[middle_row, middle_col]
                
                # Convert depth value to meters (assuming depth units are in millimeters)
                distance = depth_value / 1000.0  # Convert from millimeters to meters
                
                return distance
            else:
                return None
        else:
            return None


    def stop(self):
        if self.cap is not None:
            self.cap.release()
        if self.depth_pipeline:
            self.depth_pipeline.stop()
            print("[INFO] Stop webcam")
        



