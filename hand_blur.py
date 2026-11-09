"""Blur detected middle-finger gestures with a short release delay."""
import cv2
import numpy as np


class HandBlurFilter:
    def __init__(self, hold_seconds=.6):
        self.hold_seconds = hold_seconds
        self.active = {}

    def update(self, result, gestures, timestamp_ms, frame_shape, mirror=False):
        now = timestamp_ms/1000
        height,width = frame_shape[:2]
        self.active = {key:value for key,value in self.active.items() if value[0]>=now}
        censored = set()
        for index,points in enumerate(result.hand_landmarks):
            side = result.handedness[index][0].category_name
            xy = np.array([[(1-p.x if mirror else p.x)*width,p.y*height] for p in points])
            low,high = xy.min(axis=0),xy.max(axis=0)
            padding = max(16,float(np.max(high-low))*.25)
            x1,y1 = np.floor(low-padding).astype(int)
            x2,y2 = np.ceil(high+padding).astype(int)
            box = (max(0,x1),max(0,y1),min(width,x2),min(height,y2))
            if gestures[index] == '중지 올리기':
                self.active[side] = (now+self.hold_seconds,box)
            elif side in self.active:
                # Follow the hand during the release delay rather than blur its old location.
                self.active[side] = (self.active[side][0],box)
            if side in self.active:
                censored.add(index)
        return censored

    def apply(self, frame):
        for _,(x1,y1,x2,y2) in self.active.values():
            region = frame[y1:y2,x1:x2]
            if not region.size:
                continue
            # Downsample heavily then smooth: avoids leaving a recognizable finger outline.
            small = cv2.resize(region,(3,3),interpolation=cv2.INTER_AREA)
            expanded = cv2.resize(small,(x2-x1,y2-y1),interpolation=cv2.INTER_LINEAR)
            frame[y1:y2,x1:x2] = cv2.GaussianBlur(expanded,(0,0),sigmaX=max(8,(x2-x1)/12))
