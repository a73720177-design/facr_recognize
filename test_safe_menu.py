"""Unified numbered enrollment/unlock menu without a physical camera."""
from pathlib import Path
import tempfile
from types import SimpleNamespace as P
import unittest
from unittest.mock import Mock,patch
from recognition import run_menu
from safe_slot import SafeSlotStore


class MenuTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.args=P(camera=0,video=None,headless=False,register=None,safe_mode=None,
                    safe_number=None,face_threshold=.45,pattern_tolerance=.23,
                    no_mirror=False,max_frames=0)
        self.samples=[{'pose':{'handRight':[[0,0],[0,1]]},'preview':''}]

    def test_register_then_unlock_same_number_without_restart(self):
        sessions=[]
        def run(session):
            sessions.append(session)
            if session.safe_mode=='enroll':
                slot=SafeSlotStore(self.root/'data'/'safe_slot.json',session.safe_number)
                slot.load_for('alice');slot.save('alice',self.samples)
        with patch('recognition.ROOT',self.root), \
             patch('recognition.FaceIdentity',return_value=Mock(database={'alice':[]})), \
             patch('safe_app.run_safe',side_effect=run), \
             patch('builtins.input',side_effect=['1','2','alice','3','2','2','0']), \
             patch('builtins.print'):
            run_menu(self.args)
        self.assertEqual([s.safe_mode for s in sessions],['enroll','verify'])
        self.assertEqual([s.safe_number for s in sessions],[2,2])
        self.assertEqual(sessions[0].enrollment_steps,3)
        self.assertEqual(sessions[0].expected_user,'alice')
        self.assertTrue(all(s.menu_session for s in sessions))
        self.assertIsNone(self.args.safe_mode)

    def test_new_face_and_pattern_registration_are_one_menu_action(self):
        events=[]
        with patch('recognition.ROOT',self.root), \
             patch('recognition.FaceIdentity',return_value=Mock(database={})), \
             patch('recognition.enroll_face',side_effect=lambda args:(events.append(('face',args.register)) or True)), \
             patch('safe_app.run_safe',side_effect=lambda args:events.append(('pattern',args.safe_number))), \
             patch('builtins.input',side_effect=['1','3','bob','2','0']),patch('builtins.print'):
            run_menu(self.args)
        self.assertEqual(events,[('face','bob'),('pattern',3)])

    def test_canceled_face_enrollment_does_not_start_pattern(self):
        with patch('recognition.ROOT',self.root), \
             patch('recognition.FaceIdentity',return_value=Mock(database={})), \
             patch('recognition.enroll_face',return_value=False), \
             patch('safe_app.run_safe') as run, \
             patch('builtins.input',side_effect=['1','1','bob','2','0']),patch('builtins.print'):
            run_menu(self.args)
        run.assert_not_called()

    def test_empty_safe_cannot_be_unlocked(self):
        with patch('recognition.ROOT',self.root),patch('safe_app.run_safe') as run, \
             patch('builtins.input',side_effect=['2','3','0']),patch('builtins.print'):
            run_menu(self.args)
        run.assert_not_called()

    def test_other_owner_cannot_be_replaced_from_registration_menu(self):
        store=SafeSlotStore(self.root/'data'/'safe_slot.json',2)
        store.load_for('alice');store.save('alice',self.samples)
        with patch('recognition.ROOT',self.root),patch('safe_app.run_safe') as run, \
             patch('recognition.enroll_face') as enroll, \
             patch('builtins.input',side_effect=['1','2','bob','0']),patch('builtins.print'):
            run_menu(self.args)
        run.assert_not_called();enroll.assert_not_called()
        self.assertEqual(store.read()['owner'],'alice')

    def test_session_error_returns_to_menu(self):
        store=SafeSlotStore(self.root/'data'/'safe_slot.json',1)
        store.load_for('alice');store.save('alice',self.samples)
        with patch('recognition.ROOT',self.root), \
             patch('safe_app.run_safe',side_effect=RuntimeError('camera unavailable')) as run, \
             patch('builtins.input',side_effect=['2','1','0']),patch('builtins.print'):
            run_menu(self.args)
        run.assert_called_once()


if __name__=='__main__':unittest.main()
