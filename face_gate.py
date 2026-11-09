"""Stable face identification and bounded loss handling for a safe session."""
class FaceGate:
    def __init__(self,stable_seconds=1.,loss_seconds=.7):
        self.stable_seconds,self.loss_seconds = stable_seconds,loss_seconds
        self.reset()

    def reset(self):
        self.candidate = self.confirmed = None
        self.since = self.missing_since = None

    def update(self,name,now):
        if self.confirmed:
            if name == self.confirmed:
                self.missing_since = None
            elif name is not None:
                self.reset()
                self.candidate,self.since = name,now
            else:
                if self.missing_since is None:
                    self.missing_since = now
                if now-self.missing_since>=self.loss_seconds:
                    self.reset()
        elif name is None:
            self.candidate,self.since = None,None
        elif name != self.candidate:
            self.candidate,self.since = name,now
        elif now-self.since>=self.stable_seconds:
            self.confirmed = name
        return self.confirmed
