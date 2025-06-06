import torch
from torch.optim.optimizer import Optimizer

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ("Lion",)


class Lion(Optimizer):
    r"""Implements Lion algorithm.

    Addapted from https://github.com/google/automl/tree/master/lion

    The Lion - EvoLved SIgn MOmeNtum - algorithm was proposed in
    https://arxiv.org/pdf/2302.06675.pdf.
    Lion aims to be more memory efficient than Adam by only tracking momentum.

    Caveats: As detailed in the paper, Lion requires a smaller learning rate
    lr, and larger decoupled weight decay to maintain effective weight decay
    strength. Also, the gain of Lion increases with the batch size.
    Furthermore, Lion was not found to outperform AdamW on some large language
    and text/image datasets.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0))
        weight_decay: weight decay (L2 penalty) (default: 0)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Lion(model.parameters(), lr=0.001)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-4,
        betas: Betas2 = (0.9, 0.99),
        weight_decay: float = 0.0,
        gamma1 = 0.85,
        gamma2 = 0.999,
        theta = 0.999,
    ):

        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.theta = theta
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    state["step"] = 0
                    state["m_norm_t"]=0
                    state["v_norm_t"]=0
                    state["m_max_t"] =0
                state["step"] += 1

                max_gradient=torch.max(grad.abs())
                m_max_t = state["m_max_t"]
                m_max_t = self.theta * m_max_t + (1 - self.theta) * max_gradient
                m_max_hat = m_max_t / (1- self.theta ** state['step'])
                mask = grad.abs() > m_max_hat
                if mask.sum()>0:
                    grad[mask] = grad[mask]/max_gradient * m_max_hat
                state["m_max_t"] = m_max_t
                grad_norm = torch.norm(grad)
                m_norm_t, v_norm_t = state["m_norm_t"], state["v_norm_t"] #####
                m_norm_t = self.gamma1 * m_norm_t + (1- self.gamma1) * grad_norm
                v_norm_t = self.gamma2 * v_norm_t + (1- self.gamma2) * grad_norm ** 2
                m_norm_hat = m_norm_t / (1- (self.gamma1)**state['step'])
                v_norm_hat = v_norm_t / (1-self.gamma2**state['step'])
                c_norm_t = m_norm_hat / (torch.sqrt(v_norm_hat)+1e-8)
                grad = grad/grad_norm * c_norm_t
                state['m_norm_t'], state['v_norm_t'] = m_norm_t, v_norm_t


                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
