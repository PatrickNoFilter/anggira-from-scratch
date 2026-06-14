"""AdamW optimizer for pure-numpy AnggiraGPT."""

import numpy as np


class AdamW:
    """AdamW optimizer with bias correction and weight decay.

    Supports checkpoints via get_state / set_state so training can resume.

    Params: list of (param_array, grad_array) tuples.
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.1):
        self.params = params  # list of (param_ndarray, grad_ndarray)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]

    def step(self):
        """Apply AdamW update: decoupled weight decay + adaptive gradient."""
        self.t += 1
        inv_1mb1 = 1.0 - self.b1 ** self.t
        inv_1mb2 = 1.0 - self.b2 ** self.t
        for i, (param, grad) in enumerate(self.params):
            # Decoupled weight decay
            param -= self.lr * self.wd * param

            # Update biased moments (in-place to reduce temporaries)
            self.m[i] += (1.0 - self.b1) * (grad - self.m[i])
            self.v[i] += (1.0 - self.b2) * (grad * grad - self.v[i])

            # Bias-corrected update
            param -= self.lr * (self.m[i] / inv_1mb1) / \
                     (np.sqrt(self.v[i] / inv_1mb2) + self.eps)

    def get_state(self):
        """Return optimizer state dict for checkpointing."""
        return {
            't': self.t,
            'lr': self.lr,
            'b1': self.b1,
            'b2': self.b2,
            'eps': self.eps,
            'wd': self.wd,
            'm': self.m,
            'v': self.v,
        }

    def set_state(self, state):
        """Restore optimizer state from a checkpoint dict."""
        self.t = state['t']
        self.lr = state['lr']
        self.b1 = state.get('b1', 0.9)
        self.b2 = state.get('b2', 0.999)
        self.eps = state.get('eps', 1e-8)
        self.wd = state.get('wd', 0.1)
        for i, (param, _) in enumerate(self.params):
            if i < len(state['m']) and state['m'][i].shape == param.shape:
                self.m[i] = state['m'][i]
                self.v[i] = state['v'][i]
