"""Face identification and captured gesture patterns for a two-stage safe."""
import argparse
from copy import copy
from contextlib import ExitStack
import cv2
import mediapipe as mp
import numpy as np
from face_identity import FaceIdentity
from safe_slot import SafeSlotStore
from vision_utils import ROOT, download_models, create_detectors, draw_korean_text


def enroll_face(args):
    slot=SafeSlotStore(ROOT/'data'/'safe_slot.json')
    slot.migrate_legacy(ROOT/'data'/'safe_patterns')
    assigned=slot.user_slot(args.register)
    if assigned is not None:
        raise PermissionError(f'{assigned}번 금고 소유자의 얼굴은 변경할 수 없습니다. 먼저 초기화하세요')
    identity = FaceIdentity(ROOT,args.face_threshold)
    with ExitStack() as stack:
        cap = cv2.VideoCapture(args.camera)
        stack.callback(cap.release)
        stack.callback(cv2.destroyAllWindows)
        if not cap.isOpened():
            raise RuntimeError('카메라를 열 수 없습니다')
        window = 'face-enrollment'
        cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE)
        cv2.setWindowTitle(window,'금고 사용자 얼굴 등록')
        notice = '한 명만 얼굴을 가까이 보여주세요'
        saved_any = False
        while True:
            ok,frame = cap.read()
            if not ok:
                raise RuntimeError('카메라 영상 수신이 중단되었습니다')
            label,feature,_ = identity.identify_person(frame)
            if not args.no_mirror:
                frame = cv2.flip(frame,1)
            lines = [f'얼굴 등록: {args.register}',label,notice,'S: 저장 | Q / Esc: 종료']
            display = cv2.copyMakeBorder(frame,150,0,0,0,cv2.BORDER_CONSTANT,value=(24,28,35))
            draw_korean_text(display,[(text,(12,8+i*32),(240,240,240)) for i,text in enumerate(lines)],17)
            cv2.imshow(window,display)
            key=cv2.waitKey(1)&0xff
            if key in (ord('s'),ord('S')):
                with slot.face_registration(args.register):
                    saved=identity.enroll(args.register,feature)
                saved_any = saved_any or saved
                notice='저장 완료. 각도를 조금 바꿔 추가 저장할 수 있습니다' if saved else '저장 실패: 한 명의 얼굴을 충분히 크게 보여주세요'
            if key in (ord('q'),ord('Q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                break
        return saved_any


def run_menu(args):
    """One terminal entry point for numbered safe enrollment and unlocking."""
    from safe_app import run_safe,prompt_slot,prompt_steps
    while True:
        print('\n=== 얼굴·동작 인증 금고 ===')
        print('1. 금고 등록 (필요하면 얼굴 등록부터 진행)')
        print('2. 금고 잠금 해제')
        print('0. 프로그램 종료')
        try:
            choice=input('작업 번호: ').strip()
        except EOFError:
            return
        if choice=='0':return
        if choice not in ('1','2'):
            print('0, 1, 2 중 하나를 입력하세요.')
            continue
        try:
            store=SafeSlotStore(ROOT/'data'/'safe_slot.json')
            store.migrate_legacy(ROOT/'data'/'safe_patterns')
            number=prompt_slot(store)
            session=copy(args)
            session.safe_number=number
            session.safe_mode='enroll' if choice=='1' else 'verify'
            session.menu_session=True
            session.expected_user=None
            if choice=='1':
                name=input('등록할 사용자 이름 (취소: 빈 입력): ').strip()
                if not name:continue
                assigned=store.user_slot(name)
                owner=store.read_all()[number]['owner']
                if assigned not in (None,number):
                    print(f'{name}님은 이미 {assigned}번 금고를 사용 중입니다.')
                    continue
                if owner not in (None,name):
                    print(f'{number}번은 다른 사용자의 금고입니다. 소유자가 먼저 초기화해야 합니다.')
                    continue
                # Collect all registration settings before opening either camera UI.
                session.enrollment_steps=prompt_steps()
                session.expected_user=name
                identity=FaceIdentity(ROOT,args.face_threshold)
                if name not in identity.database:
                    print('새 사용자입니다. 얼굴 화면에서 S로 저장하고 Q로 다음 단계로 이동하세요.')
                    face_args=copy(session)
                    face_args.register=name
                    if not enroll_face(face_args):
                        print('얼굴을 저장하지 않아 금고 등록을 취소했습니다.')
                        continue
                print(f'{number}번 등록: 얼굴 인증 후 각 단계에서 P를 누르세요.')
            elif store.read_all()[number]['owner'] is None:
                print(f'{number}번은 비어 있습니다. 먼저 등록 메뉴를 이용하세요.')
                continue
            else:
                print(f'{number}번 잠금 해제: 얼굴 확인 후 U로 동작 인증을 시작하세요.')
            run_safe(session)
        except (RuntimeError,OSError,ValueError,cv2.error) as error:
            print(f'작업을 중단하고 메뉴로 돌아갑니다: {error}')


def self_test():
    identity=FaceIdentity(ROOT)
    blank=np.zeros((480,640,3),np.uint8)
    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=blank)
    with ExitStack() as stack:
        pose,hand=create_detectors(stack,download_models())
        assert not pose.detect_for_video(image,0).pose_landmarks
        assert not hand.detect_for_video(image,0).hand_landmarks
        assert identity.identify_person(blank)[1] is None
    print('OK: YuNet/SFace initialization and pose/hand blank-frame inference')


def main():
    parser=argparse.ArgumentParser(description='얼굴 식별 + 동작 패턴 2단계 인증 금고')
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--video',help='인증 처리 점검용 영상 파일')
    parser.add_argument('--headless',action='store_true',help='화면 없이 처리 점검')
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--no-mirror',action='store_true')
    parser.add_argument('--register',help='등록할 사용자 이름. S로 얼굴 저장')
    parser.add_argument('--safe-mode',choices=('enroll','verify'))
    parser.add_argument('--safe-number',type=int,choices=range(1,SafeSlotStore.SLOT_COUNT+1),help='사용할 금고 번호 1~3 (생략 시 터미널 선택)')
    parser.add_argument('--face-threshold',type=float,default=.45)
    parser.add_argument('--pattern-tolerance',type=float,default=.23)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if not 0<args.face_threshold<=1 or not .05<=args.pattern_tolerance<=.5:
        parser.error('얼굴 기준은 0 초과 1 이하, 패턴 오차는 0.05~0.5여야 합니다')
    if args.max_frames<0:
        parser.error('--max-frames는 0 이상이어야 합니다')
    if args.register is not None:
        if not args.register.strip() or args.video or args.headless or args.safe_mode or args.self_test:
            parser.error('얼굴 등록에는 이름과 웹캠 화면이 필요하며 다른 모드와 함께 실행할 수 없습니다')
        args.register=args.register.strip()
        enroll_face(args)
    elif args.self_test:
        self_test()
    else:
        if args.safe_mode is None and not args.video and not args.headless:
            run_menu(args)
        else:
            from safe_app import run_safe
            run_safe(args)


if __name__=='__main__':
    try:
        main()
    except (RuntimeError,OSError,ValueError,cv2.error) as error:
        raise SystemExit(f'오류: {error}') from error
