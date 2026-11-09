"""Safe-oriented UI reusing the existing landmark detectors."""
from contextlib import ExitStack
import time
from types import SimpleNamespace
import cv2
import mediapipe as mp
from vision_utils import ROOT, download_models, create_detectors, draw_korean_text, korean_font, draw, HAND_EDGES, EDGES
from safe_pattern import PatternSafe, describe_pose
from face_identity import FaceIdentity
from face_gate import FaceGate
from safe_slot import SafeSlotStore


def wrap_panel_lines(lines,width,size=17):
    font = korean_font(size)
    wrapped = []
    for line in lines:
        current = ''
        for char in line:
            if current and font.getlength(current+char)>width-24:
                wrapped.append(current)
                current = ''
            current += char
        wrapped.append(current)
    return wrapped


def prompt_steps():
    while True:
        try:
            value = input('등록할 동작 단계 수 (1~8): ').strip()
        except EOFError:
            raise RuntimeError('단계 수 입력이 취소되었습니다')
        if value.isdecimal() and 1<=int(value)<=8:
            return int(value)
        print('1~8 사이 정수를 입력하세요.')


def prompt_slot(store):
    print('\n[금고 번호]')
    for number,data in store.read_all().items():
        print(f'{number}번: '+('사용 중' if data['owner'] else '비어 있음'))
    while True:
        try:value=input('사용할 금고 번호 (1~3): ').strip()
        except EOFError:raise RuntimeError('금고 번호를 입력하거나 --safe-number 옵션을 지정하세요')
        if value in ('1','2','3'):return int(value)
        print('1~3 중 하나를 입력하세요.')


def run_safe(args):
    interactive = not args.video and not args.headless
    mode = args.safe_mode or 'verify'
    if mode=='enroll' and not interactive:
        raise RuntimeError('패턴 등록은 터미널 입력과 웹캠 화면이 필요합니다')
    slot = SafeSlotStore(ROOT/'data'/'safe_slot.json')
    slot.migrate_legacy(ROOT/'data'/'safe_patterns')
    number=getattr(args,'safe_number',1)
    if number is None:number=prompt_slot(slot) if interactive else 1
    slot=SafeSlotStore(slot.path,number)
    steps = (getattr(args,'enrollment_steps',None) or prompt_steps()) if mode=='enroll' else 3
    expected_user=getattr(args,'expected_user',None)
    identity = FaceIdentity(ROOT,args.face_threshold)
    slot_revision = slot.read()['revision']
    reset_until = 0.
    if not identity.database:
        raise RuntimeError('등록된 얼굴이 없습니다. --register "이름"으로 먼저 얼굴을 등록하세요')
    gate = FaceGate()
    safe = None
    current_user = None
    paths = download_models()
    with ExitStack() as stack:
        pose,hand = create_detectors(stack,paths)
        cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
        stack.callback(cap.release)
        stack.callback(cv2.destroyAllWindows)
        if not cap.isOpened():
            raise RuntimeError('카메라 또는 영상 파일을 열 수 없습니다')
        window = 'pattern-safe'
        if not args.headless:
            cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE)
            cv2.setWindowTitle(window,f'{number}번 금고 · 얼굴·동작 인증')
        previous,count = -1,0
        while True:
            ok,frame = cap.read()
            if not ok:
                if safe: safe.lock()
                if args.video:
                    break
                raise RuntimeError('카메라 입력이 중단되어 잠금 상태로 종료합니다')
            now = time.monotonic()
            timestamp = max(previous+1,int(now*1000))
            previous = timestamp
            image = mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            slot_data = slot.read()
            if slot_data['revision'] != slot_revision:
                if safe: safe.lock()
                safe,current_user = None,None
                gate.reset()
                reset_until = 0.
                slot_revision = slot_data['revision']
            identity_label,_,name = identity.identify_person(frame)
            confirmed = gate.update(name,now)
            if confirmed != current_user:
                if safe: safe.lock()
                safe = None
                current_user = confirmed
                reset_until = 0.
                assigned=slot.user_slot(confirmed)
                if (confirmed and slot_data['owner'] in (None,confirmed) and assigned in (None,number)
                        and expected_user in (None,confirmed)):
                    safe = PatternSafe(slot.path,
                                       steps,5.,args.pattern_tolerance,owner=confirmed,storage=slot)
                    if mode=='enroll' and not safe.pattern:
                        safe.start_registration(now)
            face_present = confirmed is not None and name==confirmed
            # The second factor is not evaluated before the first factor passes.
            if face_present and safe is not None:
                poses = pose.detect_for_video(image,timestamp)
                hands = hand.detect_for_video(image,timestamp)
            else:
                poses = SimpleNamespace(pose_landmarks=[])
                hands = SimpleNamespace(handedness=[],hand_landmarks=[])
            descriptor = describe_pose(hands,poses,(frame.shape[1],frame.shape[0]))
            # Store a clean camera image and compare before drawing any overlays.
            registration_finished=False
            if safe and face_present:
                previous_state=safe.state
                safe.update(descriptor,frame,now)
                registration_finished=(previous_state=='registering' and safe.state=='locked' and bool(safe.pattern))
                # Our own completed enrollment updates the stored revision too.
                slot_revision = slot.expected_revision
                slot_data = slot.read()
            elif safe:
                safe.recent.clear()
            if not face_present:
                reset_until = 0.
            for points in hands.hand_landmarks:
                draw(frame,points,HAND_EDGES,color=(0,220,255))
            if poses.pose_landmarks:
                draw(frame,poses.pose_landmarks[0],EDGES,visibility=True)
            if not args.video and not args.no_mirror:
                frame = cv2.flip(frame,1)
            # A separate panel prevents long instructions covering the user's pose.
            color = (100,255,140) if safe and safe.state=='unlocked' and face_present else (255,230,160)
            slot_label = '사용 중 · 소유자 전용' if slot_data['owner'] else '비어 있음 · 신규 사용자 등록 가능'
            status=' | '.join(f'{n}번: '+('사용 중' if item['owner'] else '빈 금고') for n,item in slot.read_all().items())
            assigned=slot.user_slot(current_user)
            lines = [f'{number}번 금고: '+slot_label,status,'1단계 인증: '+identity_label]
            if face_present:
                lines.append(f'{current_user}님의 사용 금고: '+(f'{assigned}번' if assigned else '미배정'))
            if safe and face_present:
                lines.append(f'1단계 통과: {current_user} | 2단계: 동작 패턴')
                lines.extend(safe.lines(now))
                if mode=='enroll' and safe.pattern and safe.state=='locked':
                    lines.append('기존 패턴 보호: U로 동작 인증 후 P로 재등록')
                if mode=='verify' and not safe.pattern:
                    lines.append('사용자 패턴 없음: 종료 후 등록 모드로 실행하세요')
                if safe.state=='unlocked':
                    lines.append('R: 소유권 초기화 · 다른 사용자에게 공간 넘기기')
                    if now<reset_until:
                        lines.append('주의: 소유자와 패턴을 지웁니다. 5초 안에 R을 다시 눌러 확정')
            elif face_present and expected_user is not None and current_user!=expected_user:
                lines.extend([f'등록 대상은 {expected_user}님입니다. 해당 사용자의 얼굴을 보여주세요',
                    '다른 사용자에게 패턴을 저장하지 않습니다'])
            elif face_present and assigned is not None and assigned!=number:
                lines.extend([f'사용자의 금고는 {assigned}번입니다. {assigned}번으로 다시 실행하세요',
                    '한 사용자당 하나의 금고만 등록할 수 있습니다','Q / Esc: 종료'])
            elif face_present and slot_data['owner'] is not None:
                lines.extend(['다른 사용자가 사용 중입니다. 등록·인증할 수 없습니다',
                    '현재 소유자가 얼굴·동작 인증 후 초기화해야 새 사용자가 등록할 수 있습니다',
                    'Q / Esc: 종료'])
            else:
                lines.extend(['금고 잠김 · 1단계 얼굴 인증 대기',
                    '등록된 사용자 한 명의 얼굴을 1초 이상 보여주세요',
                    '얼굴이 확인되기 전에는 동작 촬영·인증을 진행하지 않습니다',
                    'Q / Esc: 종료'])
            visible = []
            if hands.hand_landmarks: visible.append(f'손 {len(hands.hand_landmarks)}개')
            if 'arms' in descriptor: visible.append('팔 자세')
            lines.append('캡처 대상: '+(', '.join(visible) or '인식 대기'))
            if getattr(args,'menu_session',False):
                lines=[line.replace('Q / Esc: 종료','Q / Esc: 메뉴로 돌아가기') for line in lines]
                lines.append('M: 메뉴로 돌아가 금고 번호·작업 다시 선택')
            wrapped = wrap_panel_lines(lines,frame.shape[1])
            panel_height = 20+len(wrapped)*30
            display = cv2.copyMakeBorder(frame,panel_height,0,0,0,cv2.BORDER_CONSTANT,value=(24,28,35))
            draw_korean_text(display,[(line,(12,8+i*30),color if i==0 else (240,240,240)) for i,line in enumerate(wrapped)],size=17)
            count += 1
            if args.headless:
                if count%30==0: print(lines)
            else:
                cv2.imshow(window,display)
                key = cv2.waitKey(1)&0xff
                if (key in (ord('q'),ord('Q'),27,ord('m'),ord('M'))
                        or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1):
                    if safe: safe.lock()
                    break
                if not args.video:
                    if safe and face_present:
                        if key in (ord('p'),ord('P')) and mode=='enroll':
                            if safe.state!='registering': safe.start_registration(time.monotonic())
                            safe.trigger_capture(time.monotonic())
                        if key in (ord('u'),ord('U')): safe.start_verification(time.monotonic())
                        if key in (ord('r'),ord('R')) and safe.state=='unlocked':
                            if time.monotonic()<reset_until:
                                slot.reset(current_user,unlocked=safe.state=='unlocked')
                                safe.lock()
                                gate.reset()
                                print(f'{number}번 금고 초기화 완료. 해당 번호에 다른 사용자가 등록할 수 있습니다.')
                                break
                            reset_until=time.monotonic()+5.
                    if key in (ord('l'),ord('L')):
                        if safe: safe.lock()
                        gate.reset()
                        reset_until = 0.
                if registration_finished and getattr(args,'menu_session',False):
                    print(f'{number}번 금고 등록 완료. 메뉴로 돌아갑니다.')
                    cv2.waitKey(1200)
                    break
            if args.max_frames>0 and count>=args.max_frames:
                if safe: safe.lock()
                break
