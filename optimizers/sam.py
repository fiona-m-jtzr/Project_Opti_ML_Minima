import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization optimizer.
    Wraps any base optimizer and performs a two-step update:
      1. Perturb weights toward the sharpest ascent direction.
      2. Step the base optimizer at the perturbed weights, then restore.

    Args:
        params:          model parameters
        base_optimizer:  a torch.optim class (e.g. torch.optim.SGD)
        rho:             neighbourhood size (default 0.05)
        adaptive:        use adaptive SAM (ASAM) if True
        **kwargs:        forwarded to base_optimizer
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0, "rho must be non-negative"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Perturb weights toward gradient ascent direction (ε-sharpness ball)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)                      # climb to w + e(w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore original weights, then take a base-optimizer step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])     # back to w
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Convenience: performs both steps if a closure is provided.
        For manual two-step usage call first_step / second_step directly.
        """
        assert closure is not None, (
            "SAM requires a closure for the two-step update. "
            "See the training loop for the correct usage pattern."
        )
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        # Collect all gradients onto the same device for norm computation
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.stack(norms).norm(p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups