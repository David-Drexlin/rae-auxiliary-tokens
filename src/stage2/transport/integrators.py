import numpy as np
import torch as th
import torch.nn as nn
from torchdiffeq import odeint
from functools import partial
from tqdm import tqdm


def _is_joint_state(x):
    return isinstance(x, tuple)


def _first_tensor(x):
    return x[0] if isinstance(x, tuple) else x


def _batch_size(x):
    return _first_tensor(x).shape[0]


def _state_map(fn, *states):
    ref = states[0]
    if th.is_tensor(ref):
        return fn(*states)

    if isinstance(ref, tuple):
        assert all(isinstance(s, tuple) and len(s) == len(ref) for s in states), "State tuple structure mismatch"
        return tuple(_state_map(fn, *xs) for xs in zip(*states))

    raise TypeError(f"Unsupported state type: {type(ref)}")


def _state_randn_like(x):
    return _state_map(lambda a: th.randn_like(a), x)


class sde:
    """SDE solver class"""

    def __init__(
        self,
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
        time_dist_shift,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type
        self.time_dist_shift = time_dist_shift

    def __Euler_Maruyama_step(self, x, mean_x, t_curr, t_next, model, **model_kwargs):
        dt = t_curr - t_next
        t = th.ones(_batch_size(x), device=_first_tensor(x).device, dtype=_first_tensor(x).dtype) * t_curr

        w_cur = _state_randn_like(x)
        dw = _state_map(lambda w: w * th.sqrt(dt), w_cur)

        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)

        mean_x = _state_map(lambda xi, di: xi - di * dt, x, drift)
        x = _state_map(lambda mx, diffi, dwi: mx + th.sqrt(2 * diffi) * dwi, mean_x, diffusion, dw)
        return x, mean_x

    def __Heun_step(self, x, _, t_curr, t_next, model, **model_kwargs):
        dt = t_curr - t_next
        t_curr_vec = th.ones(_batch_size(x), device=_first_tensor(x).device, dtype=_first_tensor(x).dtype) * t_curr
        t_next_vec = th.ones(_batch_size(x), device=_first_tensor(x).device, dtype=_first_tensor(x).dtype) * t_next

        w_cur = _state_randn_like(x)
        dw = _state_map(lambda w: w * th.sqrt(dt), w_cur)

        diffusion = self.diffusion(x, t_curr_vec)
        xhat = _state_map(lambda xi, diffi, dwi: xi + th.sqrt(2 * diffi) * dwi, x, diffusion, dw)

        K1 = self.drift(xhat, t_curr_vec, model, **model_kwargs)
        xp = _state_map(lambda xhi, k1i: xhi - dt * k1i, xhat, K1)
        K2 = self.drift(xp, t_next_vec, model, **model_kwargs)

        x_next = _state_map(lambda xhi, k1i, k2i: xhi - 0.5 * dt * (k1i + k2i), xhat, K1, K2)
        return x_next, xhat

    def __forward_fn(self):
        sampler_dict = {
            "euler": self.__Euler_Maruyama_step,
            "heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except Exception:
            raise NotImplementedError("Sampler type not implemented.")

        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init
        samples = []
        sampler = self.__forward_fn()

        for t_curr, t_next in zip(self.t[:-1], self.t[1:]):
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, t_curr, t_next, model, **model_kwargs)
                samples.append(x)

        return samples


class ode:
    """ODE solver class"""

    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        time_dist_shift,
    ):
        assert t0 != t1, "ODE sampler requires a non-degenerate time interval"

        self.drift = drift
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type
        
    def sample(self, x, model, **model_kwargs):
        device = _first_tensor(x).device

        def _fn(t, x):
            if isinstance(x, tuple):
                t_vec = th.ones(x[0].size(0), device=device, dtype=x[0].dtype) * t
            else:
                t_vec = th.ones(x.size(0), device=device, dtype=x.dtype) * t
            model_output = self.drift(x, t_vec, model, **model_kwargs)
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]

        samples = odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol,
        )
        return samples