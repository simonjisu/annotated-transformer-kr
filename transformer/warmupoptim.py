class WarmUpOptim(object):
    """Varied the Learning Rate"""
    def __init__(self, warmup_steps, d_model, optimizer):
        """
        ref: http://nlp.seas.harvard.edu/2018/04/03/attention.html
        """
        self._optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self._step = 0
        self._lrate = 0
        
    def step(self):
        """Update parameters and rate"""
        self._step += 1
        lrate = self.cal_rate()
        for p in self._optimizer.param_groups:
            p["lr"] = lrate
        self._lrate = lrate
        self._optimizer.step()
        
    def zero_grad(self):
        """zero_grad for optimizer"""
        return self._optimizer.zero_grad()
    
    def cal_rate(self, step=None):
        """lrate = d_model**(-0.5) * min(step**(-0.5), step*warmup_steps**(-1.5))"""
        if step is None:
            step = self._step
            
        lrate = self.d_model**(-0.5) * min(step**(-0.5), step*self.warmup_steps**(-1.5))
        return lrate