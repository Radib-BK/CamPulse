
import numpy as np
import cv2
import dlib
from imutils import face_utils
import imutils
import pyrealsense2 as rs

class FaceDetection(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.fa = face_utils.FaceAligner(self.predictor, desiredFaceWidth=256)
        self.pipeline = rs.pipeline()
        self.pipeline.start()

    def face_detect(self):
        frame = None
        face_frame = np.zeros((10, 10, 3), np.uint8)
        mask = np.zeros((10, 10, 3), np.uint8)
        ROI1 = np.zeros((10, 10, 3), np.uint8)
        ROI2 = np.zeros((10, 10, 3), np.uint8)
        status = False

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            # Convert the color frame to a numpy array
            frame = np.asanyarray(color_frame.get_data())
            # Flip the image horizontally for correct display
            frame = cv2.flip(frame, 1)

            if frame is None:
                return frame, face_frame, ROI1, ROI2, status, mask

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale image
            rects = self.detector(gray, 0)

            if len(rects) > 0:
                status = True
                (x, y, w, h) = face_utils.rect_to_bb(rects[0])

                if y < 0:
                    print("a")
                    return frame, face_frame, ROI1, ROI2, status, mask

                face_frame = frame[y:y+h, x:x+w]

                if face_frame.shape[:2][1] != 0:
                    face_frame = imutils.resize(face_frame, width=256)

                grayf = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
                rectsf = self.detector(grayf, 0)

                if len(rectsf) > 0:
                    shape = self.predictor(grayf, rectsf[0])
                    shape = face_utils.shape_to_np(shape)

                    for (a, b) in shape:
                        cv2.circle(face_frame, (a, b), 1, (0, 0, 255), -1)

                    cv2.rectangle(face_frame, (shape[54][0], shape[29][1]),
                                  (shape[12][0], shape[33][1]), (0, 255, 0), 0)
                    cv2.rectangle(face_frame, (shape[4][0], shape[29][1]),
                                  (shape[48][0], shape[33][1]), (0, 255, 0), 0)

                    ROI1 = face_frame[shape[29][1]:shape[33][1],
                                      shape[54][0]:shape[12][0]]

                    ROI2 = face_frame[shape[29][1]:shape[33][1],
                                      shape[4][0]:shape[48][0]]

                    rshape = np.zeros_like(shape)
                    rshape = self.face_remap(shape)
                    mask = np.zeros((face_frame.shape[0], face_frame.shape[1]))

                    cv2.fillConvexPoly(mask, rshape[0:27], 1)

        return frame, face_frame, ROI1, ROI2, status, mask

    def face_remap(self, shape):
        remapped_image = shape.copy()
        # left eye brow
        remapped_image[17] = shape[26]
        remapped_image[18] = shape[25]
        remapped_image[19] = shape[24]
        remapped_image[20] = shape[23]
        remapped_image[21] = shape[22]
        # right eye brow
        remapped_image[22] = shape[21]
        remapped_image[23] = shape[20]
        remapped_image[24] = shape[19]
        remapped_image[25] = shape[18]
        remapped_image[26] = shape[17]
        # neatening
        remapped_image[27] = shape[0]

        remapped_image = cv2.convexHull(shape)
        return remapped_image
