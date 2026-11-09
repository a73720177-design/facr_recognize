"""Exclusive ownership, reset authorization and stale-session regression tests."""
import json
from pathlib import Path
import tempfile
import unittest
from safe_slot import SafeSlotStore
from safe_pattern import PatternSafe
from types import SimpleNamespace as P
from unittest.mock import patch,Mock
import numpy as np


class SlotTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'safe_slot.json'
        self.store=SafeSlotStore(self.path)
        self.samples=[{'pose':{'handRight':[[0.,0.],[0.,1.]]},'preview':''}]

    def claim(self,owner='alice'):
        self.store.load_for(owner)
        self.store.save(owner,self.samples)

    def test_first_complete_pattern_claims_single_space(self):
        self.assertIsNone(self.store.read()['owner'])
        safe=PatternSafe(self.path,steps=1,owner='alice',storage=self.store)
        safe.start_registration(0)
        safe.trigger_capture(0)
        self.assertIsNone(self.store.read()['owner'])
        frame=np.zeros((100,100,3),np.uint8)
        for t in (4.6,4.8,5): safe.update(self.samples[0]['pose'],frame,t)
        self.assertEqual(self.store.read()['owner'],'alice')
        self.assertEqual(safe.state,'locked')

    def test_other_user_cannot_load_save_or_reset(self):
        self.claim()
        other=SafeSlotStore(self.path)
        before=self.path.read_bytes()
        with self.assertRaises(PermissionError):other.load_for('bob')
        with self.assertRaises(PermissionError):other.save('bob',self.samples)
        with self.assertRaises(PermissionError):other.reset('bob',unlocked=True)
        self.assertEqual(self.path.read_bytes(),before)

    def test_owner_needs_unlocked_state_to_reset(self):
        self.claim()
        with self.assertRaises(PermissionError):self.store.reset('alice')
        self.assertEqual(self.store.read()['owner'],'alice')

    def test_reset_frees_space_for_new_user_and_revokes_previous(self):
        self.claim()
        old=SafeSlotStore(self.path)
        old.load_for('alice')
        self.store.reset('alice',unlocked=True)
        data=self.store.read()
        self.assertIsNone(data['owner'])
        self.assertEqual(data['samples'],[])
        with self.assertRaises(PermissionError):old.save('alice',self.samples)
        with self.assertRaises(PermissionError):old.reset('alice',unlocked=True)
        new=SafeSlotStore(self.path)
        self.assertIsNone(new.load_for('bob'))
        new.save('bob',self.samples)
        self.assertEqual(new.read()['owner'],'bob')

    def test_parallel_enrollment_cannot_overwrite_first_owner(self):
        other=SafeSlotStore(self.path)
        self.store.load_for('alice')
        other.load_for('bob')
        self.store.save('alice',self.samples)
        with self.assertRaises(PermissionError):other.save('bob',self.samples)

    def test_restart_preserves_owner(self):
        self.claim()
        restarted=SafeSlotStore(self.path)
        self.assertEqual(restarted.read()['owner'],'alice')
        with self.assertRaises(PermissionError):restarted.load_for('bob')
        self.assertEqual(len(restarted.load_for('alice')['samples']),1)

    def test_numbered_spaces_have_independent_owners_and_reset(self):
        self.claim('alice')
        second=SafeSlotStore(self.path,2)
        second.load_for('bob');second.save('bob',self.samples)
        self.assertEqual(second.user_slot('alice'),1)
        self.assertEqual(second.user_slot('bob'),2)
        self.store.reset('alice',unlocked=True)
        self.assertIsNone(self.store.read()['owner'])
        self.assertEqual(second.read()['owner'],'bob')
        self.assertEqual(second.load_for('bob')['samples'],self.samples)

    def test_one_user_cannot_own_two_spaces_even_after_parallel_start(self):
        second=SafeSlotStore(self.path,2)
        self.store.load_for('alice');second.load_for('alice')
        self.store.save('alice',self.samples)
        with self.assertRaises(PermissionError):second.save('alice',self.samples)
        with self.assertRaises(PermissionError):second.load_for('alice')
        self.assertIsNone(second.read()['owner'])

    def test_existing_single_space_becomes_number_one(self):
        self.path.write_text(json.dumps({'version':1,'revision':'legacy',
            'owner':'alice','samples':self.samples}))
        second=SafeSlotStore(self.path,2)
        self.assertEqual(second.user_slot('alice'),1)
        self.assertIsNone(second.read()['owner'])
        second.load_for('bob');second.save('bob',self.samples)
        self.assertEqual(self.store.read()['samples'],self.samples)
        self.assertEqual(json.loads(self.path.read_text())['version'],2)

    def test_new_face_allowed_but_existing_owner_face_protected(self):
        self.claim('alice')
        with self.store.face_registration('bob'):pass
        with self.assertRaises(PermissionError):
            with self.store.face_registration('alice'):pass

    def test_invalid_number_rejected(self):
        for number in (0,5,-1):
            with self.assertRaises(ValueError):SafeSlotStore(self.path,number)

    def test_number_prompt_rejects_invalid_choices(self):
        from safe_app import prompt_slot
        with patch('builtins.input',side_effect=['0','5','abc','3']),patch('builtins.print'):
            self.assertEqual(prompt_slot(self.store),3)

    def test_face_registration_blocked_until_reset(self):
        with self.store.face_registration():pass
        self.claim()
        with self.assertRaises(PermissionError):
            with self.store.face_registration():pass
        self.store.reset('alice',unlocked=True)
        with self.store.face_registration():pass

    def test_lock_prevents_concurrent_changes(self):
        self.store.load_for('alice')
        with self.store.transaction():
            with self.assertRaises(RuntimeError):self.store.save('alice',self.samples)
        self.assertFalse(self.path.with_suffix('.lock').exists())

    def test_corrupt_slot_does_not_become_vacant(self):
        self.path.write_text(json.dumps({'version':1,'owner':'alice','revision':'x','samples':[]}))
        with self.assertRaises(ValueError):self.store.read()

    def test_legacy_owner_migrates_once_and_reset_stays_empty(self):
        directory=Path(self.temp.name)/'legacy'
        directory.mkdir()
        (directory/'old.json').write_text(json.dumps({'version':2,'owner':'alice','samples':self.samples}))
        self.assertTrue(self.store.migrate_legacy(directory))
        self.assertEqual(self.store.read()['owner'],'alice')
        self.store.reset('alice',unlocked=True)
        self.assertFalse(self.store.migrate_legacy(directory))
        self.assertIsNone(self.store.read()['owner'])
        self.assertTrue((directory/'old.json').exists())

    def test_multiple_legacy_owners_require_explicit_choice(self):
        directory=Path(self.temp.name)/'legacy'
        directory.mkdir()
        for owner in ('alice','bob'):
            (directory/f'{owner}.json').write_text(json.dumps({'version':2,'owner':owner,'samples':self.samples}))
        with self.assertRaises(RuntimeError):self.store.migrate_legacy(directory)
        self.assertFalse(self.path.exists())

    def test_application_blocks_second_factor_for_non_owner(self):
        from safe_app import run_safe
        with tempfile.TemporaryDirectory() as root:
            store=SafeSlotStore(Path(root)/'data'/'safe_slot.json')
            store.load_for('alice');store.save('alice',self.samples)
            identity=Mock(database={'alice':[],'bob':[]})
            identity.identify_person.return_value=('bob',None,'bob')
            cap=Mock()
            cap.isOpened.return_value=True
            cap.read.return_value=(True,np.zeros((240,640,3),np.uint8))
            pose,hand=Mock(),Mock()
            args=P(safe_mode='verify',video='test.avi',headless=True,pattern_tolerance=.23,
                   face_threshold=.45,camera=0,no_mirror=False,max_frames=3)
            with patch('safe_app.ROOT',Path(root)),patch('safe_app.FaceIdentity',return_value=identity), \
                 patch('safe_app.download_models',return_value={}), \
                 patch('safe_app.create_detectors',return_value=(pose,hand)), \
                 patch('safe_app.cv2.VideoCapture',return_value=cap), \
                 patch('safe_app.time.monotonic',side_effect=[0.,1.1,1.2]):
                run_safe(args)
            hand.detect_for_video.assert_not_called()
            pose.detect_for_video.assert_not_called()
            self.assertEqual(store.read()['owner'],'alice')


if __name__=='__main__':unittest.main()
