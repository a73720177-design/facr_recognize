"""Tests for timed enrollment, persistence and fail-closed verification."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from types import SimpleNamespace as P
from safe_pattern import PatternSafe, compare_pose, describe_pose
from face_gate import FaceGate
from unittest.mock import patch


class PatternTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'safe.json'
        self.safe = PatternSafe(self.path,steps=2)
        self.frame = np.zeros((120,160,3),np.uint8)
        self.a = {'handRight':[[0.,0.],[0.,1.]]}
        self.b = {'handRight':[[0.,0.],[1.,0.]]}

    def capture(self,pose,deadline):
        if self.safe.state=='registering':
            self.safe.trigger_capture(deadline-5)
        for time in (deadline-.4,deadline-.2,deadline):
            self.safe.update(pose,self.frame,time)

    def enroll(self):
        self.safe.start_registration(0)
        self.capture(self.a,5)
        self.capture(self.b,10)

    def test_countdown_and_ordered_success(self):
        self.safe.start_registration(0)
        self.safe.update(self.a,self.frame,4)
        self.assertEqual(self.safe.index,0)
        self.capture(self.a,5)
        self.assertEqual(self.safe.index,1)
        self.assertFalse(self.path.exists())
        self.capture(self.b,10)
        self.assertTrue(self.path.exists())
        self.assertEqual(self.safe.state,'locked')
        self.safe.start_verification(11)
        self.capture(self.a,16)
        self.assertEqual(self.safe.state,'verifying')
        self.capture(self.b,21)
        self.assertEqual(self.safe.state,'unlocked')
        self.safe.lock()
        self.assertEqual(self.safe.state,'locked')

    def test_wrong_order_locks_and_restarts(self):
        self.enroll()
        self.safe.start_verification(11)
        self.capture(self.b,16)
        self.assertEqual(self.safe.state,'locked')
        self.assertEqual(self.safe.index,0)

    def test_missing_tracking_never_unlocks(self):
        self.enroll()
        self.safe.start_verification(11)
        self.safe.update({},self.frame,16)
        self.assertEqual(self.safe.state,'locked')

    def test_jitter_rejected(self):
        self.safe.start_registration(0)
        self.safe.trigger_capture(0)
        for t,p in [(4.6,self.a),(4.8,self.b),(5,self.a)]:
            self.safe.update(p,self.frame,t)
        self.assertEqual(self.safe.index,0)
        self.assertIsNone(self.safe.deadline)

    def test_cancel_does_not_replace_saved_pattern(self):
        self.enroll()
        old=self.path.read_bytes()
        self.safe.state='unlocked'
        self.safe.start_registration(20)
        self.capture(self.b,25)
        self.safe.lock()
        self.assertEqual(self.path.read_bytes(),old)

    def test_locked_registration_is_blocked(self):
        self.enroll()
        self.safe.start_registration(20)
        self.assertEqual(self.safe.state,'locked')

    def test_restart_loads_pattern_but_stays_locked(self):
        self.enroll()
        restarted=PatternSafe(self.path)
        self.assertEqual(restarted.state,'locked')
        self.assertEqual(len(restarted.pattern),2)
        self.assertTrue(json.loads(self.path.read_text())['samples'][0]['preview'])

    def test_missing_or_extra_group_rejected(self):
        self.assertFalse(compare_pose(self.a,{})[0])
        self.assertFalse(compare_pose(self.a,{**self.a,'handLeft':[[0,0]]})[0])
        self.assertFalse(compare_pose({}, {})[0])
        self.assertFalse(compare_pose(self.a,{'handRight':[[float('nan'),0],[0,1]]})[0])
        self.assertFalse(compare_pose({'expression':[0]*5},{'expression':[0]*5})[0])

    def test_hand_descriptor_translation_scale_invariance(self):
        pts=[P(x=.3+i*.01,y=.3+i*.012) for i in range(21)]
        def describe(points):
            return describe_pose(P(handedness=[[P(category_name='Left')]],hand_landmarks=[points]),
                P(pose_landmarks=[]),(640,480))
        translated=[P(x=p.x*1.3+.1,y=p.y*1.3+.1) for p in pts]
        self.assertTrue(compare_pose(describe(pts),describe(translated))[0])

    def test_no_detection_and_duplicate_hands_rejected(self):
        self.assertEqual(describe_pose(P(handedness=[],hand_landmarks=[]),
            P(pose_landmarks=[]),(640,480)),{})
        self.assertEqual(describe_pose(P(handedness=[[P(category_name='Left')]]*2,
            hand_landmarks=[[]]*2),P(pose_landmarks=[]),(640,480)),{})

    def test_each_enrollment_step_waits_for_p(self):
        self.safe.start_registration(0)
        self.safe.update(self.a,self.frame,100)
        self.assertEqual(self.safe.index,0)
        self.capture(self.a,105)
        self.safe.update(self.b,self.frame,200)
        self.assertEqual(self.safe.index,1)
        self.assertIsNone(self.safe.deadline)

    def test_owner_mismatch_rejected(self):
        self.enroll()
        with self.assertRaises(ValueError):
            PatternSafe(self.path,owner='someone-else')

    def test_legacy_expression_pattern_is_not_loaded(self):
        self.path.write_text(json.dumps({'version':1,'samples':[{'pose':{'expression':[0]*5}}]}))
        with self.assertRaises(ValueError): PatternSafe(self.path)


class GateTests(unittest.TestCase):
    def test_face_enrollment_still_saves_on_s(self):
        from recognition import enroll_face
        from unittest.mock import Mock
        args=P(camera=0,face_threshold=.45,no_mirror=False,register='alice')
        cap=Mock()
        cap.isOpened.return_value=True
        cap.read.side_effect=[(True,np.zeros((240,640,3),np.uint8)) for _ in range(2)]
        identity=Mock()
        identity.identify_person.return_value=('face',np.array([1.,0.]),None)
        identity.enroll.return_value=True
        with tempfile.TemporaryDirectory() as folder, \
             patch('recognition.ROOT',Path(folder)), \
             patch('recognition.FaceIdentity',return_value=identity), \
             patch('recognition.cv2.VideoCapture',return_value=cap), \
             patch('recognition.cv2.namedWindow'),patch('recognition.cv2.setWindowTitle'), \
             patch('recognition.cv2.imshow'),patch('recognition.cv2.getWindowProperty',return_value=1), \
             patch('recognition.cv2.waitKey',side_effect=[ord('s'),ord('q')]):
            enroll_face(args)
        self.assertEqual(identity.enroll.call_count,1)
        self.assertEqual(identity.enroll.call_args.args[0],'alice')
        cap.release.assert_called_once()

    def test_identity_required_and_stable(self):
        gate=FaceGate()
        self.assertIsNone(gate.update(None,0))
        self.assertIsNone(gate.update('alice',1))
        self.assertIsNone(gate.update('alice',1.5))
        self.assertEqual(gate.update('alice',2),'alice')

    def test_changed_user_and_loss_revoke(self):
        gate=FaceGate()
        gate.update('alice',0);gate.update('alice',1)
        self.assertIsNone(gate.update('bob',1.1))
        self.assertEqual(gate.update('bob',2.2),'bob')
        gate.update(None,3)
        self.assertIsNone(gate.update(None,3.8))

    def test_steps_prompt_rejects_bad_input(self):
        from safe_app import prompt_steps
        with patch('builtins.input',side_effect=['0','9','abc','4']),patch('builtins.print'):
            self.assertEqual(prompt_steps(),4)

    def test_camera_opens_after_registration_step_prompt(self):
        from safe_app import run_safe
        from unittest.mock import Mock
        args=P(safe_mode='enroll',video=None,headless=False,pattern_steps=3,
               pattern_tolerance=.23,face_threshold=.45,camera=0)
        events=[]
        identity=Mock(database={'alice':[]})
        with tempfile.TemporaryDirectory() as folder, \
             patch('safe_app.ROOT',Path(folder)), \
             patch('safe_app.prompt_steps',side_effect=lambda:(events.append('steps') or 4)), \
             patch('safe_app.FaceIdentity',return_value=identity), \
             patch('safe_app.download_models',return_value={}), \
             patch('safe_app.create_detectors',return_value=(None,None)), \
             patch('safe_app.cv2.VideoCapture',side_effect=lambda *a:(events.append('camera') or Mock(isOpened=lambda:False))):
            with self.assertRaises(RuntimeError): run_safe(args)
        self.assertEqual(events,['steps','camera'])

    def test_app_skips_second_factor_before_face_confirmation(self):
        from safe_app import run_safe
        from unittest.mock import Mock
        args=P(safe_mode='verify',video='test.avi',headless=True,pattern_steps=3,
               pattern_tolerance=.23,face_threshold=.45,camera=0,no_mirror=False,max_frames=3)
        cap=Mock()
        cap.isOpened.return_value=True
        cap.read.return_value=(True,np.zeros((240,640,3),np.uint8))
        identity=Mock(database={'alice':[]})
        identity.identify_person.return_value=('alice',None,'alice')
        pose=Mock();pose.detect_for_video.return_value=P(pose_landmarks=[])
        hand=Mock();hand.detect_for_video.return_value=P(handedness=[],hand_landmarks=[])
        with tempfile.TemporaryDirectory() as folder, \
             patch('safe_app.ROOT',Path(folder)), \
             patch('safe_app.FaceIdentity',return_value=identity), \
             patch('safe_app.download_models',return_value={}), \
             patch('safe_app.create_detectors',return_value=(pose,hand)), \
             patch('safe_app.cv2.VideoCapture',return_value=cap), \
             patch('safe_app.time.monotonic',side_effect=[0.,1.1,1.2]):
            run_safe(args)
        self.assertEqual(identity.identify_person.call_count,3)
        self.assertEqual(hand.detect_for_video.call_count,2)
        self.assertEqual(pose.detect_for_video.call_count,2)


if __name__=='__main__':
    unittest.main()
