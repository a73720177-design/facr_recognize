"""Snapshot pose descriptors and timed, ordered safe-pattern verification."""
import base64
from collections import deque
import json
import math
from pathlib import Path

import cv2
import numpy as np

def describe_pose(hands, poses, image_size):
    """Use landmarks, not names such as rock or scissors, as a pose password."""
    width,height = image_size
    groups = {}
    sides = [categories[0].category_name for categories in hands.handedness]
    if len(sides) != len(set(sides)):
        return {}  # Ambiguous hands should not become a password sample.
    for side,points in zip(sides,hands.hand_landmarks):
        xy = np.array([[p.x*width,p.y*height] for p in points],dtype=float)
        scale = max(np.linalg.norm(xy[0]-xy[9]),np.linalg.norm(xy[5]-xy[17]))
        if scale < 20 or not np.isfinite(xy).all():
            return {}
        groups['hand'+side] = ((xy-xy[0])/scale).tolist()
    if poses.pose_landmarks:
        points = poses.pose_landmarks[0]
        ids = (11,12,13,14,15,16)
        if all(points[i].visibility >= .65 for i in ids):
            xy = np.array([[points[i].x*width,points[i].y*height] for i in ids])
            scale = np.linalg.norm(xy[0]-xy[1])
            if scale >= 35 and np.isfinite(xy).all():
                groups['arms'] = ((xy-(xy[0]+xy[1])/2)/scale).tolist()
    return groups


def compare_pose(expected, actual, tolerance=.23):
    # A lost hand/arm/face must not accidentally make matching easier.
    if not expected or set(expected) != set(actual):
        return False, None
    errors = []
    for key in expected:
        a,b = np.asarray(expected[key]),np.asarray(actual[key])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            return False,None
        if key not in ('handLeft','handRight','arms') or a.ndim!=2 or a.shape[1]!=2:
            return False,None
        error = float(np.sqrt(np.mean(np.sum((a-b)**2,axis=-1))))
        errors.append(error)
    error = max(errors)
    return error <= tolerance,error


class PatternSafe:
    def __init__(self,path,steps=3,delay=5.,tolerance=.23,owner=None,storage=None):
        self.path = Path(path)
        self.steps,self.delay,self.tolerance = steps,delay,tolerance
        self.owner = owner
        self.storage = storage
        self.pattern = []
        self.state = 'locked'
        self.message = '패턴을 등록하거나 잠금 해제를 시작하세요'
        self.samples = []
        self.index = 0
        self.deadline = 0.
        self.recent = deque()
        data = storage.load_for(owner) if storage is not None else (
            json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else None)
        if data is not None:
            if data.get('version') != 2 or data.get('owner')!=owner or not 1<=len(data.get('samples',[]))<=8:
                raise ValueError('패턴 파일 형식이 올바르지 않습니다')
            self.pattern = data['samples']
            for item in self.pattern:
                if not compare_pose(item['pose'],item['pose'],self.tolerance)[0]:
                    raise ValueError('패턴 특징 데이터가 올바르지 않습니다')

    def start_registration(self,now):
        if self.state in ('registering','verifying'):
            return
        if self.pattern and self.state != 'unlocked':
            self.message = '기존 패턴으로 잠금을 해제한 뒤 재등록하세요'
            return
        self._start('registering',now)
        self.deadline = None
        self.message = '자세를 준비하고 P를 누르면 5초 뒤 촬영합니다'

    def trigger_capture(self,now):
        if self.state=='registering' and self.deadline is None:
            self.deadline = now+self.delay
            self.recent.clear()
            self.message = '5초 뒤 자동 촬영합니다. 자세를 유지하세요'

    def start_verification(self,now):
        if self.state in ('registering','verifying','unlocked'):
            return
        if not self.pattern:
            self.message = '먼저 P 키로 패턴을 등록하세요'
            return
        self._start('verifying',now)
        self.message = '등록했던 자세를 같은 순서로 보여주세요'

    def _start(self,state,now):
        self.state,self.index,self.samples = state,0,[]
        self.deadline = now+self.delay
        self.recent.clear()

    def lock(self):
        self.state,self.samples,self.index = 'locked',[],0
        self.recent.clear()
        self.message = '잠김 상태입니다. 진행 중인 입력은 취소되었습니다'

    def update(self,pose,frame,now):
        if self.state not in ('registering','verifying'):
            return
        if self.deadline is None:
            return
        if not pose or (self.recent and set(self.recent[-1][1]) != set(pose)):
            self.recent.clear()
        if pose:
            self.recent.append((now,pose))
        while self.recent and now-self.recent[0][0] > .7:
            self.recent.popleft()
        if now < self.deadline:
            return
        stable = len(self.recent)>=3 and now-self.recent[0][0]>=.20
        if stable:
            median = {key:np.median([p[key] for _,p in self.recent],axis=0).tolist() for key in pose}
            stable = all(compare_pose(median,p,self.tolerance*.8)[0] for _,p in self.recent)
        if not stable:
            if self.state == 'verifying':
                self.lock()
                self.message = '인식 불안정 또는 자세 미검출: 잠금을 유지합니다. U로 재시도'
            else:
                self.deadline = None
                self.message = '자세가 불안정합니다. P를 눌러 같은 단계를 재촬영하세요'
                self.recent.clear()
            return
        if self.state == 'registering':
            # Embed a small capture preview in the same atomic local file.
            h,w = frame.shape[:2]
            preview = cv2.resize(frame,(240,max(1,round(h*240/w))))
            ok,encoded = cv2.imencode('.jpg',preview,[cv2.IMWRITE_JPEG_QUALITY,75])
            if not ok:
                raise RuntimeError('패턴 캡처 이미지 저장 실패')
            self.samples.append({'pose':median,'preview':base64.b64encode(encoded).decode('ascii')})
            self.index += 1
            if self.index == self.steps:
                if self.storage is not None:
                    self.storage.save(self.owner,self.samples)
                else:
                    self.path.parent.mkdir(parents=True,exist_ok=True)
                    temporary = self.path.with_suffix('.tmp')
                    temporary.write_text(json.dumps({'version':2,'owner':self.owner,'samples':self.samples},ensure_ascii=False),encoding='utf-8')
                    temporary.replace(self.path)
                self.pattern = self.samples.copy()
                self.lock()
                self.message = f'{len(self.pattern)}단계 패턴 저장 완료. U 키로 잠금 해제를 시작하세요'
                return
            self.message = f'{self.index}단계 캡처 완료. 다음 자세를 준비하고 P를 누르세요'
        else:
            matched,error = compare_pose(self.pattern[self.index]['pose'],median,self.tolerance)
            if not matched:
                self.lock()
                self.message = '패턴 불일치: 잠금을 유지합니다. U 키로 처음부터 재시도'
                return
            self.index += 1
            if self.index == len(self.pattern):
                self.state = 'unlocked'
                self.message = '모든 동작이 일치합니다. 잠금 해제 완료'
                self.recent.clear()
                return
            self.message = '동작 일치. 다음 순서의 자세를 준비하세요'
        self.deadline = None if self.state=='registering' else now+self.delay
        self.recent.clear()

    def lines(self,now):
        status = {'locked':'잠김','unlocked':'잠금 해제','registering':'패턴 등록 중','verifying':'패턴 확인 중'}[self.state]
        lines = [f'동작 패턴 금고 · {status}',self.message]
        if self.state in ('registering','verifying'):
            total = self.steps if self.state=='registering' else len(self.pattern)
            countdown = 'P 키로 촬영 시작' if self.deadline is None else f'자동 캡처까지 {max(0,math.ceil(self.deadline-now))}초'
            lines.append(f'단계 {self.index+1} / {total} · {countdown}')
            lines.append('촬영 직전에는 자세를 잠시 유지하세요')
        else:
            lines.append(f'저장된 패턴: {len(self.pattern)}단계 | 신규 등록: {self.steps}단계')
        lines.extend(['P: 현재 단계 촬영 시작  U: 동작 인증 시작  L: 잠그기 / 취소','Q / Esc: 종료  |  실제 금고 장치는 연결되지 않았습니다'])
        return lines
