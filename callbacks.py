from torch import inf


class EarlyStopping:
    def __init__(self, monitor='val_loss', mode='min', patience=30, verbose=True) -> None:
        self.monitor = 'val_loss'
        self.max_patience = patience
        self.mode = mode
        self.verbose = verbose
        if mode == 'min':
            self.score = inf
        elif mode == 'max':
            self.score = 0.
        self.patience = 0

    def __call__(self, score):
        if self.mode == 'min':
            if score < self.score:
                self.patience = 0
                self.score = score
            else:
                self.patience += 1
        elif self.mode == 'max':
            if score > self.score:
                self.patience = 0
                self.score = score
            else:
                self.patience += 1

    def state_dict(self):
        return {'monitor': self.monitor, 'max_patience': self.max_patience, 'mode': self.mode, 'score': self.score, 'patience': self.patience}

    def load_state_dict(self, state: dict):
        for k, v in state.items():
            st = getattr(self, k)
            st = v
    