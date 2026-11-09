"""Local face enrollment and matching using OpenCV YuNet and SFace."""
import json
from pathlib import Path
import urllib.request
import cv2
import numpy as np

MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


class FaceIdentity:
    def __init__(self, root, threshold=.45):
        root = Path(root)
        directory = root / "models"
        directory.mkdir(exist_ok=True)
        for name,url in MODEL_URLS.items():
            target = directory / name
            if not target.exists():
                temporary = target.with_suffix('.tmp')
                try:
                    print(f"Downloading {name}...")
                    urllib.request.urlretrieve(url, temporary)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        self.detector = cv2.FaceDetectorYN.create(str(directory / 'face_detection_yunet_2023mar.onnx'), '', (320,320), .8)
        self.recognizer = cv2.FaceRecognizerSF.create(str(directory / 'face_recognition_sface_2021dec.onnx'), '')
        self.threshold = threshold
        self.database_path = root / 'data' / 'faces.json'
        self.database = {}
        if self.database_path.exists():
            self.database = json.loads(self.database_path.read_text(encoding='utf-8'))

    def identify_person(self, frame):
        """Return UI label and enrollment feature; reject multiple/too-small faces."""
        h,w = frame.shape[:2]
        self.detector.setInputSize((w,h))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return '얼굴 미검출', None, None
        if len(faces) != 1:
            return '한 명만 화면에 보여주세요', None, None
        face = faces[0]
        if min(face[2:4]) < 80:
            return '얼굴을 더 가까이 보여주세요', None, None
        aligned = self.recognizer.alignCrop(frame, face)
        feature = self.recognizer.feature(aligned).reshape(-1)
        feature /= max(float(np.linalg.norm(feature)), 1e-8)
        if not self.database:
            return '등록된 얼굴 없음', feature, None
        scores = []
        for name, samples in self.database.items():
            score = max(float(np.dot(feature,np.asarray(sample,dtype=np.float32))) for sample in samples)
            scores.append((score,name))
        scores.sort(reverse=True)
        score,name = scores[0]
        # Close matches between different names are left unknown.
        ambiguous = len(scores)>1 and score-scores[1][0]<.05
        if score < self.threshold or ambiguous:
            return f'미등록 / 판별 불가 (유사도 {score:.2f})', feature, None
        return f'{name} (유사도 {score:.2f})', feature, name

    def enroll(self, name, feature):
        name = name.strip()
        if not name or feature is None:
            return False
        samples = self.database.setdefault(name, [])
        samples.append(feature.tolist())
        self.database[name] = samples[-10:]
        self.database_path.parent.mkdir(exist_ok=True)
        temporary = self.database_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.database,ensure_ascii=False),encoding='utf-8')
        temporary.replace(self.database_path)
        return True
