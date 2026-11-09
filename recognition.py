"""Webcam/video face, finger joints, expressions and simple body actions."""
import argparse
from contextlib import ExitStack
from pathlib import Path
from functools import lru_cache
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
from face_identity import FaceIdentity, analyze_face
from hand_motion import HandMotionTracker
from hand_blur import HandBlurFilter

ROOT = Path(__file__).resolve().parent
WINDOW_NAME = "recognition"
WINDOW_TITLE = "test1"


def open_display_window():
    """Open a window at the original video frame size."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_TITLE)


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
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}
HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
              (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
              (15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
# Landmark IDs: wrist, then four landmarks for each finger, thumb to pinky.
HAND_JOINT_NAMES = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP"]
for finger in ("INDEX", "MIDDLE", "RING", "PINKY"):
    HAND_JOINT_NAMES.extend(f"{finger}_{joint}" for joint in ("MCP", "PIP", "DIP", "TIP"))


def finger_angles(world):
    """Estimated 3D joint angles in degrees; a straight joint is about 180."""
    return {name: angle(world[a], world[b], world[c])
            for name, a, b, c in [("엄지",2,3,4),("검지",5,6,7),
                                  ("중지",9,10,11),("약지",13,14,15),
                                  ("소지",17,18,19)]}


def recognize_eye_action(scores):
    """Only call a wink when the other eye is clearly open."""
    left = scores.get("eyeBlinkLeft", 0.)
    right = scores.get("eyeBlinkRight", 0.)
    if left >= .55 and right >= .55:
        return "양쪽 눈 감기"
    if left >= .55 and right <= .30:
        return "왼쪽 눈만 감기"
    if right >= .55 and left <= .30:
        return "오른쪽 눈만 감기"
    return None


def recognize_hand_gesture(world, landmarks=None, image_size=None):
    """Combine image geometry with 3D estimates to handle front-facing hands."""
    if len(world) != 21:
        return "판별 불가"
    coordinates = np.array([[p.x,p.y,p.z] for p in world], dtype=float)
    projected = None
    if landmarks is not None and image_size is not None and len(landmarks) == 21:
        width, height = image_size
        # Pixel coordinates prevent the camera aspect ratio distorting angles.
        projected = np.array([[p.x*width,p.y*height] for p in landmarks], dtype=float)

    def distance(points, a, b):
        return float(np.linalg.norm(points[a]-points[b]))

    def thumb_is_open():
        for points in (projected, coordinates):
            if points is None:
                continue
            palm = max(distance(points,5,17), distance(points,0,9))
            length = distance(points,2,3)+distance(points,3,4)
            if (palm > 1e-8 and length > palm*.20 and
                    distance(points,2,4)/max(length,1e-8) >= .80 and
                    distance(points,4,5)/palm >= .40):
                return True
        return False

    def finger_state(points, mcp):
        pip, dip, tip = mcp+1, mcp+2, mcp+3
        segments = [distance(points,a,b) for a,b in ((mcp,pip),(pip,dip),(dip,tip))]
        length = sum(segments)
        palm = max(distance(points,0,9), distance(points,5,17))
        if palm < 1e-8 or min(segments) < palm*.025 or length < palm*.35:
            return "unknown"  # Too foreshortened or degenerate to classify.
        reach = distance(points,0,tip)/max(distance(points,0,pip),1e-8)
        straightness = distance(points,mcp,tip)/length
        if straightness >= .86 and reach >= 1.10:
            return "open"
        if reach <= .95 or (straightness <= .72 and reach <= 1.08):
            return "closed"
        return "unknown"

    states = []
    for mcp in (5,9,13,17):
        state = finger_state(projected,mcp) if projected is not None else "unknown"
        if state == "unknown":
            state = finger_state(coordinates,mcp)
        pip, dip, tip = mcp+1, mcp+2, mcp+3
        p = angle(world[mcp], world[pip], world[dip])
        d = angle(world[pip], world[dip], world[tip])
        if state == "unknown":
            if p >= 145 and d >= 140 and distance(coordinates,mcp,tip) > 1e-8:
                state = "open"
            elif p <= 115 or d <= 115:
                state = "closed"
        states.append(state)
    if states == ["closed", "open", "closed", "closed"]:
        return "중지 올리기"
    if states == ["closed"] * 4:
        # Evaluate thumbs-up before rock, since both have four curled fingers.
        # Image y grows downward, so an upward thumb has a negative y vector.
        if projected is not None:
            palm = max(distance(projected,0,9), distance(projected,5,17))
            thumb_length = distance(projected,2,3)+distance(projected,3,4)
            direction = projected[4]-projected[2]
            upward = -direction[1]
            folded_top = min(projected[i,1] for i in (5,6,9,10,13,14,17,18))
            if (palm > 1e-8 and thumb_length > palm*.35 and
                    distance(projected,2,4)/max(thumb_length,1e-8) >= .85 and
                    upward > palm*.35 and upward >= abs(direction[0])*.70 and
                    projected[4,1] < folded_top-palm*.10):
                return "엄지 올리기"
        return "바위"
    if states == ["open", "open", "closed", "closed"]:
        return "가위"
    # Also accept the thumb/index scissors variant shown by the user.
    if states == ["open", "closed", "closed", "closed"] and thumb_is_open():
        return "가위"
    if states == ["open"] * 4 and thumb_is_open():
        return "보자기"
    return "판별 불가"


def draw_hands(frame, result, mirror=False, show_ids=True, motion_labels=None, censored_indices=()):
    """Draw in display coordinates so labels stay readable after mirroring."""
    from types import SimpleNamespace
    h, w = frame.shape[:2]
    summaries = []
    text_entries = []
    for index, points in enumerate(result.hand_landmarks):
        handedness = result.handedness[index][0]
        side = handedness.category_name
        color = (255,160,40) if side == "Left" else (255,80,220)
        display = [SimpleNamespace(x=1-p.x if mirror else p.x, y=p.y) for p in points]
        draw(frame, display, HAND_EDGES, color=color)
        for joint_id, p in enumerate(display):
            x, y = int(p.x*w), int(p.y*h)
            if show_ids:
                cv2.putText(frame, str(joint_id), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX,
                            .4, (0,0,0), 3)
                cv2.putText(frame, str(joint_id), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX,
                            .4, color, 1)
        wrist = display[0]
        side_label = "왼손" if side == "Left" else "오른손" if side == "Right" else "손"
        if index in censored_indices:
            summaries.append(f'{side_label}: 부적절한 손 동작 감지 · 블러 처리')
            continue
        gesture = recognize_hand_gesture(result.hand_world_landmarks[index], points, (w,h))
        label = f"{side_label}: {gesture}"
        position = (max(0,min(w-180,int(wrist.x*w))), max(0,min(h-26,int(wrist.y*h)+10)))
        text_entries.append((label, position, color))
        angles = finger_angles(result.hand_world_landmarks[index])
        summaries.append(f"{side_label}: {gesture} ({handedness.score:.0%})")
        if motion_labels is not None:
            summaries.append(f'움직임: {motion_labels[index]}')
        summaries.append(" | ".join(f"{name} {value:.0f}°" for name,value in angles.items()))
    if text_entries:
        draw_korean_text(frame, text_entries)
    return summaries
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


def angle(a, b, c):
    u = np.array([a.x-b.x, a.y-b.y, a.z-b.z])
    v = np.array([c.x-b.x, c.y-b.y, c.z-b.z])
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    if denominator < 1e-8:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v)/denominator, -1, 1))))


def recognize_arm_action(landmarks):
    if not landmarks:
        return "신체 미검출"
    labels = []
    for shoulder, wrist, side in [(11,15,"왼팔"),(12,16,"오른팔")]:
        if min(landmarks[shoulder].visibility, landmarks[wrist].visibility) > .6:
            if landmarks[wrist].y < landmarks[shoulder].y - .08:
                labels.append(f"{side} 들기")
    return " / ".join(labels) or "기본 자세"


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
    face = stack.enter_context(vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=paths["face"]),
            running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True)))
    pose = stack.enter_context(vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=paths["pose"]),
            running_mode=vision.RunningMode.VIDEO, num_poses=1)))
    hand = stack.enter_context(vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=paths["hand"]),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
            min_tracking_confidence=.5)))
    return face, pose, hand


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, help="Use a video file instead of webcam")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--hide-hand-ids", action="store_true", help="Hide hand joint numbers")
    parser.add_argument("--headless", action="store_true", help="Process without a display")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--self-test", action="store_true", help="Load models and infer a blank frame; no camera")
    parser.add_argument("--register", help="등록할 이름. 화면에서 S를 눌러 얼굴 특징을 저장")
    parser.add_argument("--face-threshold", type=float, default=.45, help="얼굴 유사도 기준 (기본 0.45)")
    args = parser.parse_args()
    if not 0 < args.face_threshold <= 1:
        parser.error('--face-threshold must be in (0, 1]')
    if args.register and (args.headless or args.video or not args.register.strip()):
        parser.error('--register requires a name and an interactive webcam')
    paths = download_models()
    identity = FaceIdentity(ROOT, args.face_threshold)
    with ExitStack() as stack:
        face, pose, hand = create_detectors(stack, paths)
        if args.self_test:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((480,640,3), np.uint8))
            assert not face.detect_for_video(image, 0).face_landmarks
            assert not pose.detect_for_video(image, 0).pose_landmarks
            assert not hand.detect_for_video(image, 0).hand_landmarks
            assert identity.identify(np.zeros((480,640,3),np.uint8))[1] is None
            print("OK: face, pose and hand models loaded; blank-frame inference passed.")
            return
        if args.video and not args.video.is_file():
            raise RuntimeError(f"Video file does not exist: {args.video}")
        source = str(args.video) if args.video else args.camera
        cap = cv2.VideoCapture(source)
        stack.callback(cap.release)
        stack.callback(cv2.destroyAllWindows)
        if not cap.isOpened():
            raise RuntimeError("Cannot open input. Check camera permissions/index or video path.")
        korean_font(18)
        if not args.headless:
            open_display_window()
        started = time.monotonic()
        previous = -1
        count = 0
        motion_tracker = HandMotionTracker()
        blur_filter = HandBlurFilter()
        registration_status = ''
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30
        while True:
            ok, frame = cap.read()
            if not ok:
                if not args.video:
                    raise RuntimeError("Camera stopped supplying frames.")
                break
            # Inference uses the original orientation so left/right stays anatomical.
            timestamp = int(count*1000/fps) if args.video else int((time.monotonic()-started)*1000)
            timestamp = max(previous+1, timestamp)
            previous = timestamp
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            faces = face.detect_for_video(image, timestamp)
            poses = pose.detect_for_video(image, timestamp)
            hands = hand.detect_for_video(image, timestamp)
            identity_label, enrollment_feature = identity.identify(frame)
            feature_lines = analyze_face(faces.face_landmarks[0], (frame.shape[1],frame.shape[0])) if faces.face_landmarks else []
            for points in faces.face_landmarks:
                draw(frame, points, color=(0,220,255))
            body = poses.pose_landmarks[0] if poses.pose_landmarks else None
            action = recognize_arm_action(body)
            if body:
                draw(frame, body, EDGES, visibility=True)
            expressions = []
            if faces.face_blendshapes:
                scores = {s.category_name:s.score for s in faces.face_blendshapes[0]}
                if (scores.get("mouthSmileLeft",0)+scores.get("mouthSmileRight",0))/2 > .5:
                    expressions.append("미소")
                eye_action = recognize_eye_action(scores)
                if eye_action:
                    expressions.append(eye_action)
                if scores.get("jawOpen",0) > .5:
                    expressions.append("입 벌리기")
            mirror = not args.video and not args.no_mirror
            if mirror:
                frame = cv2.flip(frame, 1)
            motion_labels = motion_tracker.update(hands,timestamp,(frame.shape[1],frame.shape[0]),mirror)
            gestures = [recognize_hand_gesture(hands.hand_world_landmarks[i],points,
                        (frame.shape[1],frame.shape[0])) for i,points in enumerate(hands.hand_landmarks)]
            censored = blur_filter.update(hands,gestures,timestamp,frame.shape,mirror)
            hand_lines = draw_hands(frame, hands, mirror=mirror, show_ids=not args.hide_hand_ids,
                                    motion_labels=motion_labels,censored_indices=censored)
            # Blur after all landmark overlays, before the status panel.
            blur_filter.apply(frame)
            count += 1
            expression_label = ', '.join(expressions) or ('기본 표정' if faces.face_landmarks else '얼굴 미검출')
            lines = [f"얼굴: {len(faces.face_landmarks)}명 | {expression_label}",
                     f"인물: {identity_label}",
                     *feature_lines,
                     f"동작: {action}",
                     f"손: {len(hands.hand_landmarks)}개 | 추정 관절 각도 (도)",
                     *hand_lines,
                     *(([f'등록 이름: {args.register} | S: 저장 | {registration_status}']) if args.register else []),
                     "Q / Esc: 종료"]
            draw_korean_text(frame, [(line, (12,8+i*28), (255,255,255))
                                     for i,line in enumerate(lines)])
            if args.headless:
                if count % 30 == 0:
                    print(f"Frame {count}: {lines[:-1]}")
            else:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('s'),ord('S')) and args.register:
                    saved = identity.enroll(args.register,enrollment_feature)
                    registration_status = '저장 완료' if saved else '등록 실패: 얼굴 한 명을 가까이 보여주세요'
                if key in (27,ord('q')) or cv2.getWindowProperty(WINDOW_NAME,cv2.WND_PROP_VISIBLE)<1:
                    break
            if args.max_frames > 0 and count >= args.max_frames:
                break
        print(f"Processed {count} frames")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
