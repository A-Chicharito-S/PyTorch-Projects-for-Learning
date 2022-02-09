class WarmUpOpt:

    def __init__(self, optimizer, dim, warmup_steps=4000, factor=1):
        self._step = 0
        self.factor = factor
        self.dim = dim
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer

    def updateRate(self):
        return self.factor * (self.dim ** (-0.5) *
                              min(self._step ** (-0.5), self._step * self.warmup_steps ** (-1.5)))

    def step(self):
        self._step += 1
        for p in self.optimizer.param_groups:
            p['lr'] = self.updateRate()
        self.optimizer.step()
        self.optimizer.zero_grad()

