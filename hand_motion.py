"""Temporal hand movement recognition in display coordinates."""
from collections import deque
import numpy as np


class HandMotionTracker:
    def __init__(self):
        self.histories = {}

    def update(self, result, timestamp_ms, image_size, mirror=False):
        now = timestamp_ms / 1000
        width,height = image_size
        labels = ['정지 / 움직임 없음'] * len(result.hand_landmarks)
        seen = set()
        sides = [items[0].category_name for items in result.handedness]
        for index, landmarks in enumerate(result.hand_landmarks):
            side = sides[index]
            # Ambiguous duplicate handedness cannot safely share one trajectory.
            if sides.count(side) != 1:
                self.histories.pop(side,None)
                labels[index] = '움직임 추적 대기'
                continue
            seen.add(side)
            points = np.array([[p.x*width/height,p.y] for p in landmarks])
            center = points[[0,5,9,13,17]].mean(axis=0)
            if mirror:
                center[0] = width/height-center[0]
            scale = max(float(np.linalg.norm(points[0]-points[9])),
                        float(np.linalg.norm(points[5]-points[17])), .02)
            history = self.histories.setdefault(side,deque())
            if history:
                elapsed = now-history[-1][0]
                jump = np.linalg.norm(center-history[-1][1])
                if elapsed <= 0 or elapsed > .30 or jump > max(.30,scale*3):
                    history.clear()
            history.append((now,center,scale))
            while history and now-history[0][0] > 1.5:
                history.popleft()
            if len(history)<3:
                labels[index] = '움직임 추적 중'
                continue

            # Count alternating horizontal excursions, ignoring small jitter.
            excursion = max(.035, float(np.median([s for _,_,s in history]))*.45)
            anchor = history[0][1][0]
            direction = 0
            reversals = 0
            last_turn = -float('inf')
            for moment,position,_ in history:
                delta = position[0]-anchor
                if direction == 0:
                    if abs(delta)>=excursion:
                        direction = 1 if delta>0 else -1
                        anchor = position[0]
                elif direction*(position[0]-anchor)>0:
                    anchor = position[0]
                elif direction*(position[0]-anchor)<=-excursion:
                    direction *= -1
                    anchor = position[0]
                    reversals += 1
                    last_turn = moment
            recent = [sample for sample in history if now-sample[0]<=.45]
            if len(recent)<2 or now-recent[0][0]<.12:
                continue
            displacement = recent[-1][1]-recent[0][1]
            threshold = max(.04,scale*.45)
            vertical_span = max(p[1] for _,p,_ in history)-min(p[1] for _,p,_ in history)
            horizontal_span = max(p[0] for _,p,_ in history)-min(p[0] for _,p,_ in history)
            # A wave needs at least two reversals, not just a one-way swipe.
            if reversals>=2 and now-last_turn<=.35 and horizontal_span>vertical_span*1.2:
                labels[index] = '손 흔들기'
            elif abs(displacement[0])>=threshold and abs(displacement[0])>abs(displacement[1])*1.3:
                labels[index] = '오른쪽 이동' if displacement[0]>0 else '왼쪽 이동'
            elif abs(displacement[1])>=threshold and abs(displacement[1])>abs(displacement[0])*1.3:
                labels[index] = '아래로 이동' if displacement[1]>0 else '위로 이동'
        for side in list(self.histories):
            if side not in seen:
                del self.histories[side]
        return labels
