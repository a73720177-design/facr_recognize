"""Shared camera landmark models and Korean rendering for the safe."""
from pathlib import Path
from functools import lru_cache
import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=8)
def korean_font(size):
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "malgun.ttf"
    if not font_path.is_file():
        raise RuntimeError("한글 글꼴을 찾을 수 없습니다: " + str(font_path))
    return ImageFont.truetype(str(font_path), size)

def draw_korean_text(frame, entries, size=18):
    """Render (text, position, BGR color) entries with a Korean TrueType font."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    painter = ImageDraw.Draw(image)
    font = korean_font(size)
    for text, position, color in entries:
        painter.text(position, text, font=font, fill=tuple(reversed(color)),
                     stroke_width=2, stroke_fill=(0,0,0))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

MODELS = {
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
              (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
              (15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]

EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
         (23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(28,30),(30,32)]

def download_models():
    directory = ROOT / "models"
    directory.mkdir(exist_ok=True)
    paths = {}
    for name, url in MODELS.items():
        target = directory / f"{name}_landmarker.task"
        if not target.exists():
            temporary = target.with_suffix(".tmp")
            print(f"Downloading {name} model...")
            try:
                urllib.request.urlretrieve(url, temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        paths[name] = str(target)
    return paths

def draw(frame, landmarks, edges=(), color=(0,255,0), visibility=False):
    h, w = frame.shape[:2]
    def point(i):
        return int(landmarks[i].x*w), int(landmarks[i].y*h)
    def visible(i):
        return not visibility or landmarks[i].visibility > .6
    for a, b in edges:
        if visible(a) and visible(b):
            cv2.line(frame, point(a), point(b), color, 2)
    for i in range(len(landmarks)):
        if visible(i):
            cv2.circle(frame, point(i), 1 if len(landmarks)>100 else 3, color, -1)


def create_detectors(stack, paths):
    pose = stack.enter_context(vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=paths['pose']),
            running_mode=vision.RunningMode.VIDEO, num_poses=1)))
    hand = stack.enter_context(vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=paths['hand']),
            running_mode=vision.RunningMode.VIDEO, num_hands=2)))
    return pose, hand
