import torch
import torch.nn as nn


class LBFGS:
    """
    Wraps torch.optim.LBFGS to expose a standard optimizer interface,
    so train_epoch() needs no modification beyond the is_sam branch.

    Usage: call set_batch(inputs, targets, model, criterion) once per
    batch *before* calling step().  The wrapper builds the closure
    internally.
    """

    def __init__(self, params, lr=1.0, max_iter=20, max_eval=None,
                 tolerance_grad=1e-7, tolerance_change=1e-9,
                 history_size=100, line_search_fn="strong_wolfe"):
        self._lbfgs = torch.optim.LBFGS(
            params,
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        self._model     = None
        self._criterion = None
        self._inputs    = None
        self._targets   = None
        self._last_loss = None
        self._last_outputs = None

        # Mimic the .param_groups interface that train.py reads for LR logging
        self.param_groups = self._lbfgs.param_groups

    # ------------------------------------------------------------------
    # Called once per batch from train_epoch (new helper)
    # ------------------------------------------------------------------
    def set_batch(self, inputs, targets, model, criterion):
        self._inputs    = inputs
        self._targets   = targets
        self._model     = model
        self._criterion = criterion

    # ------------------------------------------------------------------
    # Standard optimizer interface
    # ------------------------------------------------------------------
    def zero_grad(self):
        self._lbfgs.zero_grad()

    def step(self, closure=None):
        """Build the closure automatically from the stored batch."""
        def _closure():
            self._lbfgs.zero_grad()
            outputs = self._model(self._inputs)
            loss = self._criterion(outputs, self._targets)
            loss.backward()
            self._last_loss    = loss
            self._last_outputs = outputs
            return loss

        self._lbfgs.step(_closure)
        return self._last_loss

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def state_dict(self):
        return self._lbfgs.state_dict()

    def load_state_dict(self, state_dict):
        self._lbfgs.load_state_dict(state_dict)
        self.param_groups = self._lbfgs.param_groups