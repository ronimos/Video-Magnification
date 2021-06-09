# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 08:42:49 2021

@author: Avalanche
"""
import cv2
import os
import numpy as np
from tqdm import tqdm
from video_magnify import MagnifyVideo

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

        