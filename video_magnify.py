# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:21:58 2018

@author: Ron Simenhois
"""

import cv2
import numpy as np
from video_utils import VideoUtils
from video_filters import Filters
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class MagnifyVideo:

    def __init__(self, roi_type='rect', video_obj=None):
        if video_obj is None:
            video_obj = VideoUtils()
        self.video_util = video_obj
        self.video = video_obj.load_video()
        self.draw_roi()
        self.fps = video_obj.fps

    def video_magnify_color(self,
                            amplify=150.0,
                            pyramid_levels=3,
                            roi=None,
                            freq_max=None,
                            freq_min=None,
                            **kwargs):
        '''
        Magnifies color changes by getting down pyramid Gaussian presentation
        of the video and multiply the Furrier Transform presentation of the
        Gaussian presentation video.
        ---------------------------------------------------------
        Params:
            amplify - (float)
            pyramid_levels - (int)
            roi - (dict) the area to amplfy in the form of:
                         {'x1':start column,
                          'x2': end column,
                          'y1': start row,
                          'y2': end row}
        return:
            magnified_video - (munpy array) A video array with the roi section
            magnfied
        '''
        repeat = kwargs.get('repeat', False)
        video = self.video
        roi = self.roi
        if self.roi_type == 'poly':
            _video = np.zeros_like(video, dtype='float')
            for i, f in tqdm(enumerate(video), total=len(video),
                             desc='Applying ROI mask to video frames'):
                _video[i][self.mask] = f[self.mask].astype(float) / 255
        else:
            _video = video[:,
                           roi['y1']: roi['y2'],
                           roi['x1']: roi['x2'],
                           :] / 255.0

        video_data, self.pyr_shapes = Filters.gaussian_filter(_video,
                                                              pyramid_levels)
        if repeat:
            r_video_data = video_data[::-1]
            video_data = \
                np.asarray((list(video_data) + list(r_video_data)) * 3)
        video_data = Filters.temporal_bandpass_filter(video_data,
                                                      self.fps,
                                                      freq_min=freq_min,
                                                      freq_max=freq_max)
        if repeat:
            video_data = video_data[len(video)*4:len(video)*5]

        video_data *= amplify
        magnified_video = self.insert_magmify_and_reconstract(video_data=video_data, 
                                                              pyramid_levels=pyramid_levels)
        return magnified_video

    def insert_magmify_and_reconstract(self, video_data, pyramid_levels):
        '''
        Reconstruct a video numpy array back from an amplfied Gaussian
        presentation of the video and insert the amplfy section (roi)
        of each frame into the video
        ---------------------------------------------------------
        Params:
            video_data - (numpy array) - a down pyramid of Gaussian
                                         presenvation of
                                         the video (in roi)
            pyramid_levels - (int)
        return:
            magnified_video - (munpy array) A video array
        '''

        magnified_video = self.video.copy()
        length, width, heigth, channels = magnified_video.shape

        for idx in tqdm(range(length), desc='Reconstracting video'):
            frame_data = video_data[idx].copy()
            for size in reversed(self.pyr_shapes):
                frame_data = cv2.pyrUp(frame_data, dstsize=size)
            magnified_video[idx] = cv2.convertScaleAbs((magnified_video[0]+frame_data))
            magnified_video[idx][~self.mask] = self.video[idx][~self.mask]
        return magnified_video
    

    def video_magnify_motion(self, low=0.4, high=3.0, amplify=20, pyramid_levels=5):
        '''
        Magnifies motion changes by getting a list of down pyramids Laplacian 
        presentation of the video and multiply the changes between the frames 
        pyramids after butter bandpass filtering by the amplify coefficient.
        ---------------------------------------------------------
        Params:
            amplify - (float)  
            pyramid_levels - (int) 
            low - (float) low end number for the Butter banpass filter
            high - (float) high end number for the Butter banpass filter
        return:
            magnified_video - (munpy array) A video array
        '''
        
        video = self.video
        if self.roi_type=='poly':
           _video = np.zeros_like(video, dtype='float')
           for i, f in tqdm(enumerate(video), total=len(video), desc='Applying ROI mask to video frames'):
               _video[i][self.mask] = f[self.mask]
        
        lap_video_list = Filters.laplacian_filter(_video, pyramid_levels)
        filtered_frames_list=[]
        for i in range(pyramid_levels):
            filter_array=Filters.butter_bandpass_filter(self.fps, lap_video_list[i] , low, high)
            filter_array*=amplify
            filtered_frames_list.append(filter_array)
        reconsted = self.reconstruct_from_filter_frames_list(filtered_frames_list, pyramid_levels)
        amplify_video = np.clip(self.video + reconsted, 0, 255).astype(np.uint8)
        return amplify_video
    
    def reconstruct_from_filter_frames_list(self, video_pyramids_list, pyramid_levels=3):
        '''
        Reconstruct a video numpy array back from a list of down pyramids if the 
        video after amply 
        ---------------------------------------------------------
        Params:
            video_pyramids_list - (list of numpy array) - each array is a down 
            pyramid of magnfied laplacian images of different levels  
            pyramid_levels - (int) 
        return:
            magnified_video - (munpy array) A video array
        '''
        length = video_pyramids_list[0].shape[0]
        magnified_video = np.zeros_like(video_pyramids_list[-1])
        for idx in range(length):   
            up_pyr = video_pyramids_list[0][idx]
            for pyr_level in range(pyramid_levels-1):
                up_pyr = cv2.pyrUp(up_pyr) + video_pyramids_list[pyr_level + 1][idx]
            magnified_video[idx] = up_pyr
        return magnified_video

       
    def get_sub_roi(self, n_splits):
        
        roi = self.roi
        xedges = np.linspace(roi['x1'], roi['x2'], n_splits, stype=int)
        yedges = np.linspace(roi['y1'], roi['y2'], n_splits, stype=int)
        for x1, x2 in zip(xedges[:-1], xedges[1:]):
            for y1, y2 in zip(yedges[:-1], yedges[1:]):
                yield dict(x1=x1, x2=x2, y1=y1, y2=y2)

    
    def video_magnify_sub_roi(self, n_splits):
        
        for sub_roi in self.get_sub_roi:
            self.video_magnify(roi=sub_roi)
            
    def chart_pixel_intesity(self, **kwargs):
        
        import scipy.fftpack
        
        freq_min = kwargs.get('freq_min', len(self.video)/self.fps*2)        
        freq_max = kwargs.get('freq_max', len(self.video)/self.fps*5)
        title    = kwargs.get('title', '')
        roi      = kwargs.get('roi', 'get_roi')
        
        
        video = self.video[1:]
        if not roi:
            roi = self.roi
        if roi == 'get_roi':
            roi = self.video_util.get_roi(video=video)

        fig, ax = plt.subplots(nrows=5,
                               gridspec_kw={'hspace': 0.5},
                               figsize=(15, 25))
        video_roi = video[:, roi['y1']: roi['y2'], roi['x1']: roi['x2'], :]
        means = video_roi.mean(axis=(1, 2, 3))
        ax[0].plot(means)
        ax[0].set_title('''Original video average pixel intensity
                           red vertical line - initiation frame, 
                           black vertical line - first sign of slab deformation''')
        ax[0].set_xticks([])
        ax[0].vlines(13, 86.6, 87.4, color='k', linewidth=3)
        ax[0].vlines(6, 86.6, 87.4, color='r', linewidth=3)

        pyr_levels = int(np.log2(min(roi['x2']-roi['x1'], roi['y2']-roi['y1']))) -1
        video_data, _ = Filters.gaussian_filter(video=video_roi,
                                                pyramid_levels=pyr_levels)
        video_data_means = video_data.mean(axis=(1, 2, 3))
        ax[1].plot(np.convolve(video_data_means, np.ones((3))/3,mode='valid'), c='g')
        ax[1].set_title('Gaussian video, 3 frames average pixel intensity')
        ax[1].vlines(13, 88.2, 88.7, color='k', linewidth=3)
        ax[1].vlines(6, 88.2, 88.7, color='r', linewidth=3)

        video_fft = scipy.fftpack.fft(video_data, axis=0)
        
        frequencies = scipy.fftpack.fftfreq(video_data.shape[0], d=1.0/self.fps)
        mean_fft = video_fft.mean(axis=(1, 2, 3))
        bound_low = (np.abs(frequencies - freq_min)).argmin()
        bound_high = (np.abs(frequencies - freq_max)).argmin()
        
        mean_fft[:bound_low] = 0
        mean_fft[bound_high:-bound_high] = 0
        mean_fft[-bound_low:] = 0
        
        mean_fft = np.real(scipy.fft.ifft(mean_fft, axis=0))
        
        ax[3].plot(mean_fft)
        ax[3].set_title('''Reconstructed video intensity from fft frequencies (blue)''')

        frequencies, fft_signal = Filters.get_fft_signal(video_roi, self.fps)
        ax[2].bar(frequencies, fft_signal)
        fft_signal[:bound_low] = 0
        fft_signal[bound_high:-bound_high] = 0
        fft_signal[-bound_low:] = 0
        
        ax[2].bar(frequencies, fft_signal, color='r',
                  label='No noise frequencies')
        ax[2].set_title(
            '''Dominante frequencies of video average pixel intensity
               None noise frequencies (red)''')
        ax[2].set_xticks(frequencies)
        ax[2].tick_params(labelrotation=45)
        
        _ax = plt.twinx(ax[4])
        _ax.plot(video_data_means, label='original mean pixel intensity')
        _ax.plot(np.convolve(video_data_means, np.ones((3))/3,mode='valid'),
                 c='g')
        _ax.set_ylabel('original pixel intensity')
        _ax.vlines(13, ymin=88, ymax=89, color='k', linewidth=3)
        _ax.vlines(6, ymin=88, ymax=89, color='r', linewidth=3)
        mag = mean_fft * 50 + video_data_means
        mag[:7] = np.nan
        ax[4].plot(mag, c='m', linewidth=3)
        ax[4].set_ylabel('magnified pixel intensity')
        ax[4].set_title('''Reconstructed video intensity from fft frequencies
                           blue - original signal (right y-axis)
                           green - 3 frame average (right y-axis)
                           magenta - magnified signal (left y-axis)''')
        
        
        plt.suptitle(title)
        plt.savefig('Johan/Results/plot4Bertil.png')

    def get_dominant_frequencies_for_location(self, **kwargs):

        video = self.video
        roi = self.video_util.get_roi(video=video)
        video_roi = video[:, roi['y1']: roi['y2'], roi['x1']: roi['x2'], :]
        video_data, _ = Filters.gaussian_filter(video_roi, 2)
        frequencies, fft_signal = Filters.get_fft_signal(video_data, self.fps)

        return frequencies, fft_signal

    def draw_roi(self, roi_type='poly'):
        if roi_type == 'rect':
            self.roi = self.video_util.get_roi(video=self.video)
            mask = np.zeros(self.video[0].shape[:2])
            mask[self.roi['y1']:self.roi['y2'], self.roi['x1']:self.roi['x2']] = 0
            self.mask = mask
        elif roi_type == 'poly':
            self.mask = self.video_util.get_poly_mask(self.video)
            # Get bounding box for the mask:
            rows = np.any(self.mask, axis=1)
            cols = np.any(self.mask, axis=0)
            x1, x2 = np.where(cols)[0][[0, -1]]
            y1, y2 = np.where(rows)[0][[0, -1]]
            self.roi = dict(x1=x1, x2=x2, y1=y1, y2=y2)
        self.roi_type = roi_type
   
    @classmethod
    def play_video(cls, video):
        idx = 0
        vlen = len(video) - 1
        while True:
            f = video[idx].copy()
            cv2.putText(f, 'Frame {}/{}'.format(idx+1, vlen),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA)
            cv2.imshow('', f)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('f'):
                idx = min(idx+1, vlen)
            if k == ord('b'):
                idx = max(0, idx-1)
            if k == 27:  # Esc
                break
        cv2.destroyAllWindows()


def play_v(v, show_frame_n=True):
    zoom = 1
    i = 0
    l = len(v) - 1
    while True:
        f = cv2.resize(v[i], None, fx=zoom, fy=zoom)
        if show_frame_n:
            cv2.putText(f, str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5, cv2.LINE_AA)
        cv2.imshow('', f)
        k = cv2.waitKey(0)&0xFF
        if k==ord('f'):
            i = min(i+1, l)
        if k==ord('b'):
            i = max(i-1, 0)
        if k==ord('o'):
            zoom /= 1.33
        if k==ord('i'):
            zoom *=1.33
        if k==ord('s'):
            os.makedirs('Temp_video_frame_grab', exist_ok=True)
            cv2.imwrite(os.path.join('Temp_video_frame_grab', 'frame_#_{}.jpg'.format(i)), f)
        if k==27:
            break
    cv2.destroyAllWindows()
    return i
        
def transitions_to_magvify(original, 
                           magnified, 
                           mask,
                           start_transition=130, 
                           transition_period=10,
                           end_transition=None):
    transition = magnified.copy()
    alpha = 1
    for i in range(start_transition):
        transition[i] = original[i].copy()
        
    for i in range(start_transition,start_transition+transition_period):
        alpha-= (1/transition_period)
        f = cv2.addWeighted(original[i], alpha, magnified[i], 1-alpha, 1)
        transition[i][vm.mask] = cv2.GaussianBlur(f, (5,5,), 0)[mask]
        
    if end_transition:
        for i in range(end_transition, end_transition+transition_period):
            f = cv2.addWeighted(original[i], alpha, magnified[i], 1-alpha, 1)
            transition[i][vm.mask] = cv2.GaussianBlur(f, (5,5,), 0)[mask]
            alpha+= (1/transition_period)
    return transition


def add_frame_no(video, time=False, start_frame_time = 0, fps=30, color = (0,0,255)):
    new_video = video.copy()
    for i in range(len(video)):
        frame_n = str(i)
        if time and i >= start_frame_time:
            t = round((i-start_frame_time)/fps, 2) 
            frame_n = frame_n + ', Time: {} sec.'.format(t)
        cv2.putText(new_video[i], 
                    frame_n, 
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    color, 
                    2, 
                    cv2.LINE_AA)
    return new_video


def track_and_color_deformation(orig_video,
                                mag_video,
                                mask,
                                start_frame,
                                **kwards):

    threshold = kwards.get('threshold', 0.075)
    noise_threshold = kwards.get('noise_threshold', 0.5)
    gray = mag_video.astype(np.float).mean(axis=-1)
    _gray = np.abs(gray[:-1]-gray[1:])
    _gray[_gray < threshold] = 0
    _gray = np.asanyarray([cv2.GaussianBlur(f, (5, 5), 0) for f in _gray])
    _gray[:start_frame + 3] = 0

    for i in tqdm(range(_gray.shape[0])):
        _gray[i] = np.divide(_gray[i]-_gray[i].min(),
                             _gray[i].max()-_gray[i].min(),
                             out=np.zeros_like(_gray[i]),
                             where=_gray[i].max() - _gray[i].min() != 0)

    dvid = np.zeros_like(_gray)
    for i in range(_gray.shape[0]):
        dvid[i][mask] = _gray[i][mask]
    # dvid[dvid<threshold] = 0
    _dvid = np.cumsum(dvid, axis=0)
    red = np.full_like(orig_video[0, ..., 0], 255)
    _video = orig_video.copy()
    _video[1:, ..., 0] = np.where(_dvid > noise_threshold,
                                  _video[1:, ..., 0]/3,
                                  _video[1:, ..., 0]).astype(np.uint8)
    _video[1:, ..., 1] = np.where(_dvid > noise_threshold,
                                  _video[1:, ..., 1]/3,
                                  _video[1:, ..., 1]).astype(np.uint8)
    _video[1:, ..., -1] = np.where(_dvid > noise_threshold,
                                   red,
                                   _video[1:, ..., -1]).astype(np.uint8)
    for i in tqdm(range(len(orig_video))):
        _video[i][~mask] = orig_video[i][~mask]
    # play_v(_video)
    return _video


if __name__ == '__main__':
    vm = MagnifyVideo(roi_type='poly')
    vm.video = add_frame_no(vm.video,
                            time=True,
                            start_frame_time=0,
                            fps=vm.fps)
    freq, noise_frequencies = vm.get_dominant_frequencies_for_location()
    freq_min = freq[noise_frequencies > 0].max()
    epsilon = 1/(vm.fps + 1)
