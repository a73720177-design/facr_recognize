"""Numbered compartments with one owner per space and one space per user."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import uuid


class SafeSlotStore:
    SLOT_COUNT=4

    def __init__(self,path,slot_number=1):
        if slot_number not in range(1,self.SLOT_COUNT+1):
            raise ValueError('금고 번호는 1~4여야 합니다')
        self.path=Path(path)
        self.slot_number=slot_number
        self.expected_revision=None

    @staticmethod
    def _empty():
        return {'version':1,'revision':None,'owner':None,'samples':[]}

    @staticmethod
    def _validate(data):
        if (not isinstance(data,dict) or data.get('version')!=1
                or not isinstance(data.get('samples'),list)):
            raise ValueError('금고 공간 파일이 손상되었습니다. 등록을 중단합니다')
        owner=data.get('owner')
        if data.get('revision') is not None and not isinstance(data['revision'],str):
            raise ValueError('금고 변경 정보가 올바르지 않습니다')
        if owner is None:
            if data['samples']:
                raise ValueError('빈 금고 공간의 패턴 데이터가 올바르지 않습니다')
        elif (not isinstance(owner,str) or not owner.strip() or not data.get('revision')
              or not 1<=len(data['samples'])<=8):
            raise ValueError('금고 소유자 정보가 올바르지 않습니다')

    def read_all(self):
        slots={str(i):self._empty() for i in range(1,self.SLOT_COUNT+1)}
        if self.path.exists():
            data=json.loads(self.path.read_text(encoding='utf-8'))
            if data.get('version')==1:
                # The old single compartment is always safe number 1.
                slots['1']=data
            elif data.get('version')==2 and isinstance(data.get('slots'),dict):
                if set(data['slots'])!=set(slots):
                    raise ValueError('금고 번호 목록이 올바르지 않습니다')
                slots=data['slots']
            else:
                raise ValueError('금고 목록 파일이 손상되었습니다')
        owners=[]
        for entry in slots.values():
            self._validate(entry)
            if entry['owner'] is not None:owners.append(entry['owner'])
        if len(owners)!=len(set(owners)):
            raise ValueError('한 사용자에게 여러 금고가 배정되어 있습니다')
        return {int(number):entry for number,entry in slots.items()}

    def read(self):
        return self.read_all()[self.slot_number]

    def user_slot(self,owner):
        if owner is None:return None
        return next((number for number,data in self.read_all().items() if data['owner']==owner),None)

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        lock=self.path.with_suffix('.lock')
        try:
            fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError('다른 프로그램이 금고 정보를 변경 중입니다. 잠시 후 다시 시도하세요')
        try:
            os.close(fd)
            yield
        finally:
            lock.unlink(missing_ok=True)

    def _write(self,owner,samples,number=None):
        number=self.slot_number if number is None else number
        slots=self.read_all()
        data={'version':1,'revision':uuid.uuid4().hex,'owner':owner,'samples':samples}
        slots[number]=data
        registry={'version':2,'slots':{str(key):value for key,value in slots.items()}}
        temporary=self.path.with_suffix('.tmp')
        try:
            temporary.write_text(json.dumps(registry,ensure_ascii=False),encoding='utf-8')
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        if number==self.slot_number:self.expected_revision=data['revision']

    def load_for(self,owner):
        assigned=self.user_slot(owner)
        if assigned is not None and assigned!=self.slot_number:
            raise PermissionError(f'이미 {assigned}번 금고를 사용 중입니다. 다른 금고에 중복 등록할 수 없습니다')
        data=self.read()
        if data['owner'] is not None and data['owner']!=owner:
            raise PermissionError('다른 사용자가 사용 중인 금고 공간입니다')
        self.expected_revision=data['revision']
        if data['owner'] is None:
            return None
        return {'version':2,'owner':owner,'samples':data['samples']}

    def migrate_legacy(self,directory):
        """Keep a unique existing owner; never resurrect patterns after a reset."""
        if self.path.exists():
            return False
        candidates=sorted(Path(directory).glob('*.json'))
        if not candidates:
            return False
        if len(candidates)!=1:
            raise RuntimeError('기존 사용자 패턴이 여러 개입니다. 단일 공간의 소유자를 먼저 정해야 합니다')
        from safe_pattern import PatternSafe
        data=json.loads(candidates[0].read_text(encoding='utf-8'))
        owner=data.get('owner')
        if not isinstance(owner,str) or not owner.strip():
            raise ValueError('기존 패턴의 소유자를 확인할 수 없습니다')
        pattern=PatternSafe(candidates[0],owner=owner)
        with self.transaction():
            if self.path.exists():
                return False
            self._write(owner,pattern.pattern,number=1)
        return True

    def save(self,owner,samples):
        if not owner or not samples:
            raise ValueError('소유자와 완료된 패턴이 필요합니다')
        with self.transaction():
            data=self.read()
            assigned=self.user_slot(owner)
            if assigned is not None and assigned!=self.slot_number:
                raise PermissionError(f'이미 {assigned}번 금고를 사용 중입니다')
            if (data['revision']!=self.expected_revision or
                    data['owner'] is not None and data['owner']!=owner):
                raise PermissionError('금고 소유 상태가 변경되었습니다. 처음부터 다시 인증하세요')
            self._write(owner,samples)

    def reset(self,owner,unlocked=False):
        if not unlocked or not owner:
            raise PermissionError('소유자의 얼굴·동작 인증 후에만 초기화할 수 있습니다')
        with self.transaction():
            data=self.read()
            if data['owner']!=owner or data['revision']!=self.expected_revision:
                raise PermissionError('현재 소유자만 금고 공간을 초기화할 수 있습니다')
            # Keep a new revision even when vacant: stale sessions cannot reclaim it.
            self._write(None,[])

    @contextmanager
    def face_registration(self,name=None):
        with self.transaction():
            occupied=self.user_slot(name) if name is not None else next(
                (n for n,d in self.read_all().items() if d['owner'] is not None),None)
            if occupied is not None:
                raise PermissionError(f'{occupied}번 금고 소유자의 얼굴은 변경할 수 없습니다. 먼저 초기화하세요')
            yield
