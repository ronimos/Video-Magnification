    # -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 11:08:34 2018

@author: Ron Simenhois
"""
import numpy as np
import cv2
import scipy.signal
import scipy.fftpack
import pywt
from tqdm import tqdm

class Filters:
    def __init__(self, video, fps):
        self.video = video
        self.fps = fps
        self.video_data = None
    
    @classmethod
    def gaussian_filter(cls, video, pyramid_levels=3):
        """
        This function creates a NumPy array of down size pyramids of 
        all the video frames. The down size pyramids use a Gaussian 
        fill when down size. This fuction update the self object's 
        video_data - the filtered video 
        ---------------------------------------------------------
        Params:
            self - the video array to filter is in the self object
            pyramid_levels (int) - The number of down pyramids levels
        return:
            None
        """
        v_length = video.shape[0]
        gauss_pyr_shapes = [video.shape[1:3][::-1]]
        for idx in tqdm(range(v_length), desc='Applying Gaussian Filter'):
            gauss_frame = np.ndarray(shape=video[idx].shape, dtype=np.float)
            gauss_frame = video[idx].copy()
            for i in range(pyramid_levels):
                gauss_frame = cv2.pyrDown(gauss_frame)
                if idx==0:
                    gauss_pyr_shapes.append((gauss_frame.shape[:2][::-1]))
            if idx==0:
                video_data = np.ndarray(shape=(v_length,) + gauss_frame.shape, dtype=np.float)
            video_data[idx]=gauss_frame
        return video_data, gauss_pyr_shapes[:-1]
    
    @classmethod    
    def laplacian_filter(cls, video, pyramid_levels=3):
        """
        This function creates a NumPy array of the laplacian video 
        frames after doing down size pyramid. This fuction update 
        the self object's 
        video_data - the filtered video 
        ---------------------------------------------------------
        Params:
            self - the video array to filter is in the self object
            pyramid_levels (int) - The number of down pyramids levels
        return:
            a tensor list with a laplacian pyramid for every frame
        """
        def build_laplacian_pyramid(frame, pyramid_levels):
            gaussian_pyramid = [frame.copy()]
            for  i in range(pyramid_levels):
                frame = cv2.pyrDown(frame)
                gaussian_pyramid.append(frame)   
                
            laplacian_pyramid = []
            for i in range(pyramid_levels, 0, -1):
                gaussian_frame = cv2.pyrUp(gaussian_pyramid[i], dstsize=gaussian_pyramid[i-1].shape[:2][::-1])
                laplacian_frame = cv2.subtract(gaussian_pyramid[i-1],gaussian_frame)
                laplacian_pyramid.append(laplacian_frame)
            return laplacian_pyramid
            
        v_length = video.shape[0]
        
        laplacian_tensor_list=[]
        for idx in tqdm(range(v_length), desc='Cerating laplacian pyramid'):
            frame = video[idx]
            pyr = build_laplacian_pyramid(frame, pyramid_levels)
            if idx==0:                
                for i in range(pyramid_levels):
                    laplacian_tensor_list.append(np.zeros((v_length, pyr[i].shape[0], pyr[i].shape[1], 3)))
            for i in range(pyramid_levels):
                laplacian_tensor_list[i][idx] = pyr[i]

        return laplacian_tensor_list
        
        
    @classmethod
    def butter_bandpass_filter(cls, fps, pyr_level_data, lowcut=None, highcut=None, order=5):
        """
        A helper fuction to calculate a and b for the Ifilter
        in the butter_bandpass_filter fuction
        ------------------------------------------------------------
        Params:
            self - the video array to filter is in the self object
            lowcut - A scalar giving the critical frequencies. 
                    For a Butterworth filter. The lowcut / (0.5 x FPS) is the point at which 
                    the gain drops to 1/sqrt(2). 
            highcut - A scalar giving the critical frequencies. 
                    For a Butterworth filter. The highcut / (0.5 x FPS) is the point at which 
                    the gain drops to 1/sqrt(2). 
            order (int) - the filter order
        return:
            Filteres signal
        """
        if lowcut==None: lowcut = 0.045 * fps
        if highcut==None: highcut = 0.1 * fps        
        nyq = 0.5 * fps
        low = lowcut / nyq # Cutoff frequency
        high = highcut / nyq
        b, a = scipy.signal.butter(order, [low, high], btype='band')   
                
        return scipy.signal.lfilter(b, a, pyr_level_data, axis=0)
            
                    
    @classmethod
    def temporal_bandpass_filter(cls,
                                 video_data,
                                 fps,
                                 freq_min=None,
                                 freq_max=None,
                                 num_dominant_freq=7,
                                 axis=0,
                                 **kwargs):
        """
        Builds Fourier Transform presentation for each "pixel" in the video's 
        Gaussian presentation.
        ------------------------------------------------------------
        Params:
            video_data - (numpy array - float) Gaussian presentation of the original video
            fps - (int) Original videos frames per second
            freq_min - (float) low boundry frequancy for the fft.
            freq_max - (float) High
            axis -
        return:
            the real portion of the inverse fourier Transform presentation
            after the noise and out of range frequancies were removed.
        """
        frequencies = scipy.fftpack.fftfreq(video_data.shape[0], d=1.0/fps)
        fft_signal = scipy.fftpack.fft(video_data, axis=axis)
        noise_mask = kwargs.get('noise_mask', np.ones_like(frequencies))
        noise_mask = np.concatenate([noise_mask,
                                     noise_mask[:: -1]],
                                    axis=0)[:len(frequencies)]
     
        if freq_min is not None and freq_max is not None:
            bound_low = (np.abs(frequencies - freq_min)).argmin()
            bound_high = (np.abs(frequencies - freq_max)).argmin()
            
            fft_signal[:bound_low] = 0
            fft_signal[bound_high:-bound_high] = 0
            fft_signal[-bound_low:] = 0
        
        
        #fft_signal[noise_mask] = 0
        return np.real(scipy.fftpack.ifft(fft_signal, axis=0))
            
    
        '''    
        fft=scipy.fftpack.fft(tensor,axis=axis)
        frequencies = scipy.fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
        bound_low = (np.abs(frequencies - low)).argmin()
        bound_high = (np.abs(frequencies - high)).argmin()
        fft[:bound_low] = 0
        fft[bound_high:-bound_high] = 0
        fft[-bound_low:] = 0
        iff=fftpack.ifft(fft, axis=axis)
        return np.abs(iff)
        '''

    @classmethod
    def get_fft_signal(cls,
                       video_data,
                       fps,
                       num_dominant_freq=12,
                       axis=0,
                       **kwargs):

        frequencies = scipy.fftpack.fftfreq(video_data.shape[0], d=1.0/fps)
        frequencies = frequencies[frequencies >= 0]
        data = video_data.mean(axis=(1, 2, 3))
        data = np.concatenate([data[:3], data, data[-3:]], axis=0)
        mean_data = np.convolve(data, np.ones(5)/5)[5:-5]
        fft_signal = np.real(scipy.fftpack.fft(mean_data, axis=axis))
        fft_signal[0] = 0
        min_freq_val = sorted(np.abs(fft_signal),
                              reverse=True)[num_dominant_freq+1]
        dominant_freqs = np.where(fft_signal > min_freq_val,
                                  fft_signal, 0)[:len(frequencies)]
        return frequencies, dominant_freqs

    @classmethod
    def wavelets_transform(cls, data):
        scale = np.arange(0, data.shape[0])
        coef, freq = pywt.cwt(data, scale, 'morl')

    def wavelets_plot(cls, coef):
        import matplotlib.pyplot as plt
        plt.figure(1, figsize=(12, 12))
        plt.imshow(coef, camp='coolwarm', aspect='auto')

    def wavelets_plot_3D(cls, data, coef):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(1,1,1, projection='3d')
        x = np.arange(0,data.shape[0], 1)
        y = np.arange(1,129)
        x,y = np.meshgrid(x,y)
        
        ax.plot_surface(x,y, coef, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        ax.set_xlabel('Time', fontsize=20)
        ax.set_ylabel('Scale', fontsize=20)
        ax.set_zlabel('Amplitude', fontsize=20)
    
    

        