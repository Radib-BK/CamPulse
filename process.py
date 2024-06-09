import cv2
import numpy as np
import time
from face_detection import FaceDetection
from scipy import signal
from face_utilities import Face_utilities
from signal_processing import Signal_processing
from imutils import face_utils
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter

class Process(object):
    def __init__(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.buffer_size = 100
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.fd = FaceDetection()
        self.bpms = []
        self.peaks = []
        self.fu = Face_utilities()
        self.sp = Signal_processing()
        self.window_size = 2  # Size of the moving average window


    
    def smooth_signal(self, signal):
        """
        Apply a moving average filter to smooth the input signal.
        """

        # Ensure window_size is smaller than the size of the input signal
        window_size = min(self.window_size, len(signal))
        smoothed_signal = uniform_filter(signal, window_size)

        return smoothed_signal  
        

    def extractColor(self, frame):
        
        #r = np.mean(frame[:,:,0])
        g = np.mean(frame[:,:,1])
        #b = np.mean(frame[:,:,2])
        #return r, g, b
        return g
   
    def run(self):
        
        frame = self.frame_in
        ret_process = self.fu.no_age_gender_face_process(frame, "5")
        if ret_process is None:
            return False
        rects, face, shape, aligned_face, aligned_shape = ret_process

        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        if(len(aligned_shape)==68):
            cv2.rectangle(aligned_face,(aligned_shape[54][0], aligned_shape[29][1]), #draw rectangle on right and left cheeks
                    (aligned_shape[12][0],aligned_shape[33][1]), (0,255,0), 0)
            cv2.rectangle(aligned_face, (aligned_shape[4][0], aligned_shape[29][1]), 
                    (aligned_shape[48][0],aligned_shape[33][1]), (0,255,0), 0)
        else:
            cv2.rectangle(aligned_face, (aligned_shape[0][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                        (aligned_shape[1][0],aligned_shape[4][1]), (0,255,0), 0)
            
            cv2.rectangle(aligned_face, (aligned_shape[2][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                        (aligned_shape[3][0],aligned_shape[4][1]), (0,255,0), 0)
        
        for (x, y) in aligned_shape: 
            cv2.circle(aligned_face, (x, y), 1, (0, 0, 255), -1)


        ROIs = self.fu.ROI_extraction(aligned_face, aligned_shape)
        green_val = self.sp.extract_color(ROIs)

        self.frame_out = frame
        self.frame_ROI = aligned_face
        
      
        
        L = len(self.data_buffer)
        
    

        g = green_val
        
        if(abs(g-np.mean(self.data_buffer))>10 and L>99): #remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
            g = self.data_buffer[-1]
        
        self.times.append(time.time() - self.t0)
        self.data_buffer.append(g)

        #only process in a fixed-size buffer
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size
            
        processed = np.array(self.data_buffer)
        

         # Apply moving average filter to smooth the signal
        smoothed = self.smooth_signal(processed)
         # Update the samples with the smoothed signal
        self.samples = smoothed


        # start calculating after the first 10 frames
        if L == self.buffer_size:
            
            self.fps = float(L) / (self.times[-1] - self.times[0])#calculate HR using a true fps of processor of the computer, not the fps the camera provide
            even_times = np.linspace(self.times[0], self.times[-1], L)
            
            processed = signal.detrend(processed)#detrend the signal to avoid interference of light change
            interpolated = np.interp(even_times, self.times, processed) #interpolation by 1
            interpolated = np.hamming(L) * interpolated#make the signal become more periodic (advoid spectral leakage)
            norm = interpolated/np.linalg.norm(interpolated)
            raw = np.fft.rfft(norm*30)#do real fft with the normalization multiplied by 10
            
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * self.freqs
            
   
            self.fft = np.abs(raw)**2#get amplitude spectrum
        
            idx = np.where((freqs > 48) & (freqs < 180))#the range of frequency that HR is supposed to be within 
            pruned = self.fft[idx]
            pfreq = freqs[idx]
            
            self.freqs = pfreq 
            self.fft = pruned
            
            idx2 = np.argmax(pruned)#max in the range can be HR
            
            self.bpm = self.freqs[idx2]
            self.bpms.append(self.bpm)
            
            processed = self.butter_bandpass_filter(processed,0.8,3,self.fps,order = 1)
            #ifft = np.fft.irfft(raw)
        self.samples = processed # multiply the signal with 5 for easier to see in the plot
     
        return True
    
    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a


    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y 
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
