# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 08:57:32 2018

@author: Ron Simenhois
"""
import numpy as np
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

AUTO_MODE = 30
MANUAL_MODE = 0


class VideoUtils:
    
    def __init__(self, file_name=''):
        
        self.file_name = file_name
        self.drawing = False
        self.x = 0
        self.y = 0
        self.ix = 0
        self.iy = 0
        self.poly = []
        self.tpoly = []
        self.roi = dict(x1=self.x, y1=self.y, x2=self.ix, y2=self.iy)
        self.roi_set = False
        self.mask = None
               
    def load_video(self):
        """
        Invokes a TK ask for file name window to get a video file name 
        and loads a video file in a numpy array
        --------------------------------------------------------------
            params: self
            return: None
        """
        if self.file_name=='':
            Tk().withdraw()
            self.file_name = askopenfilename()
        cap = cv2.VideoCapture(self.file_name)
        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = int(round(cap.get(cv2.CAP_PROP_FPS)))
        
        video_buffer = []#np.ndarray(shape=(self.length, self.heigth, self.width, 3), dtype=np.uint8)
        for i in tqdm(range(self.length), desc='Loading video from: {}'.format(self.file_name)):
            ret, frame = cap.read()
            if not ret:
                break
            video_buffer.append(frame)
        #assert(i==self.length-1)
        video_buffer = np.array(video_buffer, dtype=np.uint8)
        self.video_buffer = video_buffer
        cap.release()
        self.ix = self.width-1
        self.iy = self.heigth-1 
        self.roi = dict(x1=self.x, y1=self.y, x2=self.ix, y2=self.iy)
        return video_buffer
    
    
    def _set_rect_roi(self, event, x, y, flags, params):
        """
        A mouse callback function that draws a rectangle on a video frame 
        and save its corners as ROI for future analasys
        --------------------------------------------------------------
            params: self
                    (int) event: mouse event (left click up, down, mouse move...)
                    (int) x, y: mouse location
                    flags: Specific condition whenever a mouse event occurs.
                            EVENT_FLAG_LBUTTON
                            EVENT_FLAG_RBUTTON
                            EVENT_FLAG_MBUTTON
                            EVENT_FLAG_CTRLKEY
                            EVENT_FLAG_SHIFTKEY
                            EVENT_FLAG_ALTKEY
                    params: user specific parameters (not used)
            return: None
        """
        if event==cv2.EVENT_LBUTTONDOWN:
            self.ix = x
            self.iy = y
            self.drawing = True
        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.x = x
                self.y = y
                self.img = self.current_frame.copy()
                cv2.rectangle(self.img,
                              (self.ix, self.iy),
                              (x, y,),
                              (0, 0, 255), 3)
        if event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def _set_poly_roi(self, event, x, y, flags, params):
        """
        A mouse callback function that draws a rectangle on a video frame
        and save its corners as ROI for future analasys
        --------------------------------------------------------------
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.tpoly.append((x, y))
            self.ix = x
            self.iy = y
            self.drawing = True
        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.img = self.current_frame.copy()
                _poly = self.tpoly + [(x, y), self.tpoly[0]]
                for p1, p2 in zip(_poly[:-1], _poly[1:]):
                    cv2.line(self.img, p1, p2, (0, 0, 255), 3)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.poly = self.tpoly
            self.tpoly = []
            self.drawing = False

    def dummy_mouse_call_back_func(self, event, x, y, flags, params):
        pass

    def play_video(self,
                   video=None,
                   window_name=None,
                   mode=MANUAL_MODE,
                   mouse_callback_func=None):
        """
        Play the video. Also alows ROI set.
        --------------------------------------------------------------
            params: self
                    (int) mode: MANUAL_MODE - 0 for flipping through the frame manualy
                                AUTO_MODE - 30 for aoutomatic play 30 ms between frames
            return: None
        """
        if mouse_callback_func==None:
            mouse_callback_func = self.dummy_mouse_call_back_func
        zoom = 1
        if video is None:
            video = self.video_buffer
        if window_name==None:
            window_name = 'Video Player'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback_func)
        idx=0
        self.current_frame = video[idx]
        self.img = self.current_frame.copy()
        while True:
            cv2.imshow(window_name, self.img)
            k = cv2.waitKey(30) & 0xFF
            if mode==MANUAL_MODE:
                if k==ord('f'):
                    idx = min(idx+1, len(video)-1)
                    self.current_frame = cv2.resize(video[idx], None, fx=zoom, fy=zoom, \
                                                    interpolation=cv2.INTER_AREA)
                    self.img = self.current_frame.copy()
                if k==ord('b'):
                    idx = max(0, idx-1)
                    self.current_frame = cv2.resize(video[idx], None, fx=zoom, fy=zoom, \
                                                    interpolation=cv2.INTER_AREA)
                    self.img = self.current_frame.copy()
                if k==ord('i'):
                    zoom *= 1.33
                    self.current_frame = cv2.resize(video[idx], None, fx=zoom, fy=zoom, \
                                                    interpolation=cv2.INTER_AREA)
                    self.img = self.current_frame.copy()
                if k==ord('o'):
                    zoom *= 0.75
                    self.current_frame = cv2.resize(video[idx], None, fx=zoom, fy=zoom, \
                                                    interpolation=cv2.INTER_AREA)
                    self.img = self.current_frame.copy()

                if k==27:
                    break
        cv2.destroyAllWindows()
        self.x /=zoom
        self.ix/=zoom
        self.y /=zoom
        self.iy/=zoom
        
        self.poly = (np.asarray(self.poly)/zoom).astype(int)
        
    def get_roi(self, video=None, window_name=None):
        """
        An interface function that get the video ROI 
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
        return:
            save_file: (str) saved video file location and name
        """
        if video is None:
            video = self.video_buffer
        self.play_video(video=video, window_name=window_name, mouse_callback_func=self._set_rect_roi)
        self.roi = dict(x1=int(min(self.x, self.ix)), x2=int(max(self.x, self.ix)),
                        y1=int(min(self.y, self.iy)), y2=int(max(self.y, self.iy)))
        self.roi_set = True
        return self.roi
    
    def get_poly_mask(self, video, window_name='get poly roi'):

        from matplotlib.path import Path

        if video is None:
            video = self.video_buffer

        self.play_video(video=video,
                        window_name=window_name,
                        mouse_callback_func=self._set_poly_roi)
        x, y = np. meshgrid(range(self.width), range(self.heigth))
        x, y = x.flatten(), y.flatten()
        roi = np.vstack((x, y)).T
        path = Path(self.poly)
        grid = path.contains_points(roi)
        self.mask = grid.reshape(self.heigth, self.width)
        rows = np.any(self.mask, axis=1)
        cols = np.any(self.mask, axis=0)
        x1, x2 = np.where(cols)[0][[0, -1]]
        y1, y2 = np.where(rows)[0][[0, -1]]
        self.roi = dict(x1=x1, x2=x2, y1=y1, y2=y2)

        return self.mask

    def save_video(self, video=None, fps=None):
        """
        Opens a tkinter save file dialog to get save file name
        and save a video under the given name
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
        return:
            save_file: (str) saved video file location and name
        """

        if video is None:
            video=self.video_buffer
        if fps==None:
            fps = self.fps
        Tk().withdraw()
        save_file = asksaveasfilename(defaultextension=".mp4")
        if save_file==None:
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename=save_file, fourcc=fourcc, fps=fps, 
                              frameSize=(self.width, self.heigth),isColor=True)
        for frame in video:
            out.write(frame)
        out.release()
        print('Video file is saved')
        return save_file
    
    @classmethod
    def _save_video(cls, video=None, fps=30):
        """
        Opens a tkinter save file dialog to get save file name
        and save a video under the given name
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
        return:
            save_file: (str) saved video file location and name
        """
        
        if video is None:
            return
        length, heigth, width, ch = video.shape
        
        Tk().withdraw()
        save_file = asksaveasfilename()
        if save_file==None:
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename=save_file, fourcc=fourcc, fps=fps, 
                              frameSize=(width, heigth),isColor=True)
        for frame in video:
            out.write(frame)
        out.release()
        print('Video file is saved')
        return save_file

        
    def trim_video(self, video=None):
        """
        Trim a video to between two specific frames
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
        return:
            trimed_video: (np.array) the trimmed video
        """
        if video is None:
            video = self.video_buffer
        start = 0
        end = self.length-1
        window = 'Click "f" for next frame, "b" for back frame, "s & e" for trimmes video start and end' 
        idx = 0
        while True:
            cv2.imshow(window, video[idx])
            k = cv2.waitKey(0) & 0xFF
            if k==ord('f'):
                idx = min(idx+1, self.length-1)
            if k==ord('b'):
                idx = max(0, idx-1)
            if k==ord('s'):
                start = idx
            if k==ord('e'):
                end = idx
            if k==27:
                break
            cv2.destroyAllWindows()
        return video[start: end, ...]
    
    def crop_video(self, video=None):
        """
        Crops all frames in a video to draw ROI
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
        return:
            croped video: (np.array) 
        """

        if video is None:
            video = self.video_buffer
        self.get_roi(video=video, window_name='Darw the ROI on any frame in the video and click Esc')
        roi = self.roi
        return video[:, roi['y1']: roi['y2'], roi['x1']: roi['x2'], :]
    
    def video_stabilizer(self, video=None):
        """
        Stabilize a video around object inside a marked ROI
        This function uses a LK oprical flow to calculate 
        movments and affin matrix to adjust the frames
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
        return:
            croped video: (np.array) 
        """
        def in_roi(roi, p):
            x, y = p
            return roi['x1'] < x < roi['x2'] and roi['y1'] < y < roi['y2']

        
        if video is None:
            video = self.video_buffer
        stab_video = np.zeros_like(video)
        roi = self.get_roi(video=video, window_name='Draw ROI to stabilize the video around it')

        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=800,
                              qualityLevel=0.01,
                              minDistance=3,
                              blockSize=3)
    
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=8,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        m_dx, m_dy = 0, 0
    
        # Take first frame and find corners in it
        old_frame = video[0]
    
        rows, cols, depth = old_frame.shape
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
        p0 = cv2.goodFeaturesToTrack(old_gray, 
                                     mask=None, 
                                     **feature_params)
        p0 = np.expand_dims([p for p in p0.squeeze() if in_roi(roi, p)], 1)# p0.copy()
        
        for idx in tqdm(range(len(video))):
    
            # Get next frame
            frame = video[idx]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            # Select good points
            try:
                good_cur = p1[np.where(st == 1)]
                good_old = p0[np.where(st == 1)]
            except TypeError as e:
                print('TypeError, no good points are avaliabole, error: {0}'.format(e))
                print('Exit video stabilizer at frame {0} out of {1}'.format(idx, self.length))
                break
            
            dx = []
            dy = []  
    
            # Draw points and calculate
            for i, (cur, old) in enumerate(zip(good_cur, good_old)):
                a, b = cur.ravel()
                c, d = old.ravel()
                dx.append(c - a)
                dy.append(d - b)
    
            m_dx += np.mean(dx)
            m_dy += np.mean(dy)
            print(m_dx,m_dy)
    
            M = np.float32([[1, 0, m_dx], [0, 1, m_dy]])
            
            stab_video[idx] = cv2.warpAffine(frame, M, (cols, rows), 
                                           cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP, 
                                           cv2.BORDER_CONSTANT).copy()

            marked = stab_video[idx].copy()
            for p in np.squeeze(good_cur):
                marked = cv2.circle(marked, tuple(p.astype(int)), 5, (255,0,0), 2)
            cv2.imshow('stab', marked)
            cv2.waitKey(0)



            # Update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_cur.reshape(-1, 1, 2)
        cv2.destroyAllWindows()
        return stab_video
    
    def track_motion(self, video=None, set_roi=False):
        """
        Track and displays the motion of items in a marked ROI. 
        This function uses a LK oprical flow to calculate 
        movments.
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
            set_roi (bool) - 
        return:
            track_video: (np.array) - the numpy array of the video with 
                                      the motion of the traked items highlited
            motion_tracker: (list) - A list of numpy arrays with the location 
                                     of the traked items for each video frame                            
        """
        
        if video is None:
            video = self.video_buffer
        if set_roi:
            roi = self.get_roi(video=video)
      
        video_track = video.copy()
        motion_tracker = []
        # Generate different colors for tracking display 
        color = np.random.randint(0,255,(100,3))
                       
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 5,
                               blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 8,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        old_gray = cv2.cvtColor(video[0], cv2.COLOR_BGR2GRAY)
        # Create mask for drawing
        mask = np.zeros_like(video[0])
        # Mask to dectate the features to track
        features_mask = np.zeros_like(old_gray)
        features_mask[roi['x1']: roi['x2'], roi['y1']: roi['y2']] = old_gray[roi['x1']: roi['x2'], roi['y1']: roi['y2']]
        # Find corners in first frame
        p0 = cv2.goodFeaturesToTrack(features_mask, mask = None, **feature_params)
        
        for idx in range(1, video.shape[0]):
            new_gray = cv2.cvtColor(video[idx], cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
            
            # Get good points
            good_old = p0[st==1]
            good_new = p1[st==1]
            motion_tracker.append(good_new)
            for i, (old, new) in enumerate(zip(good_old, good_new)):
                (ox, oy) = old.reval()
                (nx, ny) = new.ravel()
                mask = cv2.circle(mask, (nx, ny), 5, color[i].tolist(), -1)
            frame = cv2.add(video[idx], mask)
            video_track[idx] = frame
            
            cv2.imshow('frame', frame)
            if cv2.waitKey(30) & 0xFF==27:
                break
            # Updat old frames and points before checking next frame
            
            old_gray = new_gray.copy()
            p0 = p1.resapr(-1,1,2)
        cv2.destroyAllWindows()
        
        return video_track, motion_tracker
    
    @classmethod
    def get_video_from_camera(cls):
        """
        Take a video for a computer camra and fave it as a mp4 file
        ---------------------------------------------------------
        Params:
        return:
            save_file: (str) - file name of the saved file
        """
        cap = cv2.VideoCapture(0)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = 30
        save_file = None
        
        Tk().withdraw()
        save_file = asksaveasfilename(defaultextension=".mp4")
        
        if save_file!=None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename=save_file, fourcc=fourcc, fps=fps, 
                                  frameSize=(width, heigth),isColor=True)
            
            while cap.isOpened():
                ret, frame = cap.read()
                cv2.imshow('Play video', frame)
                out.write(frame)
                if cv2.waitKey(30)==27:
                    break
            cv2.destroyAllWindows()
            out.release()
            cap.release()
        return save_file
    
    @classmethod
    def play_any_video(cls, video):
        
        zoom = 1.0
        window = 'i - zoom in, o - zoom out, f - forward, b - backward, Esc to exit'
        idx = 0
        while True:
            frame = cv2.resize(video[idx], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA)
            cv2.imshow(window, frame)
            k = cv2.waitKey(0)
            if k==ord('i'):
                zoom *= 1.25
            if k==ord('o'):
                zoom /= 1.25
            if k==ord('f'):
                idx = min(idx+1, video.shape[0])
            if k==ord('b'):
                idx = max(idx-1, 0)
            if k==27:
                break
                
    @classmethod
    def chart_frequencies(cls, video, fps, bounds=None):
        
        if bounds:
            _video = video[:, bounds['y1']: bounds['y2'], bounds['x1']: bounds['x2'], :].copy()
            averages = np.array([f.mean() for f in _video])
        else:
            averages = np.array([f.mean() for f in video])
        averages = averages - np.min(averages)
        
        fig, ax = plt.subplots(ncols=1, nrows=2)
        ax[0].set_title('Average Pixel Intesity')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Brightness')
        ax[0].plot(averages)
    
        freqs = scipy.fftpack.fftfreq(len(averages), d=1.0 / fps)
        fft = np.abs(scipy.fftpack.fft(averages))
        idx = np.argsort(freqs)
    
        ax[1].set_title('FFT')
        ax[1].set_xlabel('Freq (Hz)')
        freqs = freqs[idx]
        fft = fft[idx]
    
        freqs = freqs[len(freqs) // 2 + 1:]
        fft = fft[len(fft) // 2 + 1:]
        ax[1].bar(freqs, abs(fft), width=3.0 / fps)
        plt.tight_layout()
            
        
if __name__=='__main__':   
    vu = VideoUtils()
        
      
        
    
    






            
            
        

        
        

        
            
            