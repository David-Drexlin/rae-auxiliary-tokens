import torch as th
import numpy as np
import logging
import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    NOISE = enum.auto()      # the model predicts epsilon
    SCORE = enum.auto()      # the model predicts ∇ log p(x)
    VELOCITY = enum.auto()   # the model predicts v(x)


class PathType(enum.Enum):
    """
    Which type of path to use.
    """
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


def truncated_logitnormal_sample(shape, mu, sigma, low=0.0, high=1.0):
    """
    Samples X in (0,1) with Z = logit(X) ~ Normal(mu, sigma^2), truncated so X in [low, high].
    """
    mu = th.as_tensor(mu)
    sigma = th.as_tensor(sigma)
    low = th.as_tensor(low)
    high = th.as_tensor(high)

    z_low = th.logit(low)
    z_high = th.logit(high)

    base = th.distributions.Normal(th.zeros_like(mu), th.ones_like(sigma))
    alpha = (z_low - mu) / sigma
    beta = (z_high - mu) / sigma

    cdf_alpha = base.cdf(alpha)
    cdf_beta = base.cdf(beta)

    out_shape = th.broadcast_shapes(shape, mu.shape, sigma.shape, low.shape, high.shape)
    U = th.rand(out_shape, device=mu.device, dtype=mu.dtype)
    U = cdf_alpha + (cdf_beta - cdf_alpha) * U.clamp_(0, 1)

    Z = mu + sigma * base.icdf(U)
    X = th.sigmoid(Z)
    return X.clamp(low, high)


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


def _state_assert_same_shape(a, b):
    if th.is_tensor(a):
        assert th.is_tensor(b), f"Expected tensor, got {type(b)}"
        assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
        return

    assert isinstance(a, tuple) and isinstance(b, tuple), f"State type mismatch: {type(a)} vs {type(b)}"
    assert len(a) == len(b), f"Tuple length mismatch: {len(a)} vs {len(b)}"
    for ai, bi in zip(a, b):
        _state_assert_same_shape(ai, bi)


def _state_mean_flat(x):
    if th.is_tensor(x):
        return mean_flat(x)

    assert isinstance(x, tuple), f"Unsupported state type: {type(x)}"
    out = None
    for xi in x:
        term = mean_flat(xi)
        out = term if out is None else out + term
    return out


def _sse_and_count(x, y):
    diff = (x - y).flatten(1)
    return (diff * diff).sum(dim=1), diff.shape[1]


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        aux_loss_weight,
        component_mode,
        time_dist_type,
        time_dist_shift,
        train_eps,
        sample_eps,
        dual_schedule_mode="none",
        patch_schedule=None,
        aux_schedule=None,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.aux_loss_weight = float(aux_loss_weight)
        self.component_mode = str(component_mode)
        if self.component_mode not in {"joint", "aux_only"}:
            raise ValueError(f"Unsupported component_mode={self.component_mode}")
        if self.component_mode == "aux_only" and self.model_type != ModelType.VELOCITY:
            raise NotImplementedError("component_mode='aux_only' currently supports velocity prediction only.")
        self.time_dist_type = time_dist_type
        self.time_dist_shift = float(time_dist_shift)
        if self.time_dist_shift <= 0.0:
            raise ValueError("time_dist_shift must be positive.")
        self.dual_schedule_mode = str(dual_schedule_mode)
        if self.dual_schedule_mode not in {"none", "shared_base_t", "dual_time_embed"}:
            raise ValueError(f"Unsupported dual_schedule_mode={self.dual_schedule_mode}")
        if self.dual_schedule_mode != "none" and self.model_type != ModelType.VELOCITY:
            raise NotImplementedError("Dual schedules currently support velocity prediction only.")

        def _make_schedule(name, schedule_cfg):
            cfg = {} if schedule_cfg is None else dict(schedule_cfg)
            schedule = EasyDict({})
            schedule.name = str(name)
            schedule.path_type = cfg.get("path_type", path_type)
            schedule.path_sampler = path_options[schedule.path_type]()
            schedule.time_shift = float(cfg.get("time_shift", self.time_dist_shift))
            if schedule.time_shift <= 0.0:
                raise ValueError(f"{name} time_shift must be positive.")
            schedule.train_eps = float(cfg.get("train_eps", train_eps))
            schedule.sample_eps = float(cfg.get("sample_eps", sample_eps))
            return schedule

        self.patch_schedule = _make_schedule("patch", patch_schedule)
        self.aux_schedule = _make_schedule("aux", aux_schedule)

        # Backward-compatible aliases used elsewhere in the codebase.
        self.path_sampler = self.patch_schedule.path_sampler
        self.train_eps = self.patch_schedule.train_eps
        self.sample_eps = self.patch_schedule.sample_eps

    @staticmethod
    def _apply_shift(t, shift):
        if shift == 1.0:
            return t
        return shift * t / (1 + (shift - 1) * t)

    @staticmethod
    def _invert_shift(t, shift):
        if shift == 1.0:
            return t
        denom = shift - (shift - 1) * t
        return t / denom.clamp_min(1e-8)

    @staticmethod
    def _shift_derivative(s, shift):
        if shift == 1.0:
            return th.ones_like(s)
        return shift / (1 + (shift - 1) * s) ** 2

    def _uses_dual_schedules(self, x):
        return self.dual_schedule_mode != "none" and _is_joint_state(x)

    def _uses_aux_only_mode(self, x):
        return self.component_mode == "aux_only" and _is_joint_state(x)

    def _component_schedule(self, idx):
        return self.patch_schedule if idx == 0 else self.aux_schedule

    def _model_time_input(self, x, time_info):
        if self.dual_schedule_mode == "dual_time_embed" and _is_joint_state(x):
            aux_t = time_info.aux if time_info.aux is not None else time_info.patch
            return (time_info.patch, aux_t)
        if self.dual_schedule_mode in {"shared_base_t", "none"}:
            return time_info.base
        return time_info.base

    def _model_time_from_base(self, x, base_t):
        if th.is_tensor(x):
            patch_t, patch_scale = self._component_time(base_t, self.patch_schedule)
            return EasyDict({
                "base": base_t,
                "patch": patch_t,
                "patch_scale": patch_scale,
                "aux": None,
                "aux_scale": None,
            })

        patch_t, patch_scale = self._component_time(base_t, self.patch_schedule)
        aux_t, aux_scale = self._component_time(base_t, self.aux_schedule)
        return EasyDict({
            "base": base_t,
            "patch": patch_t,
            "patch_scale": patch_scale,
            "aux": aux_t,
            "aux_scale": aux_scale,
        })

    def _component_time(self, base_t, schedule):
        if self.dual_schedule_mode == "none":
            return base_t, th.ones_like(base_t)

        base_s = self._invert_shift(base_t, self.time_dist_shift)
        eff_t = self._apply_shift(base_s, schedule.time_shift)
        d_eff_ds = self._shift_derivative(base_s, schedule.time_shift)
        d_base_ds = self._shift_derivative(base_s, self.time_dist_shift)
        scale = d_eff_ds / d_base_ds.clamp_min(1e-8)
        return eff_t, scale

    @staticmethod
    def _expand_scale_like(scale, x):
        return path.expand_t_like_x(scale, x)

    def _combine_component_losses(self, component_losses):
        loss = None
        for idx, li in enumerate(component_losses):
            weight = 1.0 if idx == 0 else self.aux_loss_weight
            weighted = li * weight
            loss = weighted if loss is None else loss + weighted
        return loss

    def _weighted_component_sse_and_count(self, model_output, xt, x0, t):
        _, drift_var = self.path_sampler.compute_drift(xt, t)
        sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))

        if self.loss_type in [WeightType.VELOCITY]:
            weight = (drift_var / sigma_t) ** 2
        elif self.loss_type in [WeightType.LIKELIHOOD]:
            weight = drift_var / (sigma_t ** 2)
        elif self.loss_type in [WeightType.NONE]:
            weight = 1
        else:
            raise NotImplementedError()

        if self.model_type == ModelType.NOISE:
            err = weight * ((model_output - x0) ** 2)
        else:
            err = weight * ((model_output * sigma_t + x0) ** 2)

        flat = err.flatten(1)
        return flat.sum(dim=1), flat.shape[1]

    def prior_logp(self, z):
        """
        Standard multivariate normal prior.
        Supports tensor or tuple state.
        """
        if th.is_tensor(z):
            shape = th.tensor(z.size(), device=z.device)
            N = th.prod(shape[1:])
            _fn = lambda x: -N / 2.0 * np.log(2 * np.pi) - th.sum(x ** 2) / 2.0
            return th.vmap(_fn)(z)

        assert isinstance(z, tuple), f"Unsupported state type: {type(z)}"
        out = None
        for zi in z:
            shape = th.tensor(zi.size(), device=zi.device)
            N = th.prod(shape[1:])
            _fn = lambda x: -N / 2.0 * np.log(2 * np.pi) - th.sum(x ** 2) / 2.0
            term = th.vmap(_fn)(zi)
            out = term if out is None else out + term
        return out

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        def _single_interval(schedule):
            t0 = 0
            t1 = 1 - 1 / 1000
            eps = schedule.train_eps if not eval else schedule.sample_eps
            sampler = schedule.path_sampler

            if type(sampler) in [path.VPCPlan]:
                t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
            elif type(sampler) in [path.ICPlan, path.GVPCPlan] and (
                self.model_type != ModelType.VELOCITY or sde
            ):
                t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
                t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

            return t0, t1

        schedules = [self.patch_schedule]
        if self.dual_schedule_mode != "none":
            schedules.append(self.aux_schedule)

        t0 = max(_single_interval(s)[0] for s in schedules)
        t1 = min(_single_interval(s)[1] for s in schedules)
        if t0 >= t1:
            raise ValueError("Dual scheduler intervals do not overlap.")

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1):
        """
        Sampling x0 & t based on shape of x1.
        Supports tensor or tuple state.
        """
        if self._uses_aux_only_mode(x1):
            x0 = (x1[0], _state_randn_like(x1[1]))
        else:
            x0 = _state_randn_like(x1)

        dist_options = self.time_dist_type.split("_")
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        bsz = _batch_size(x1)

        if dist_options[0] == "uniform":
            t = th.rand((bsz,), device=_first_tensor(x1).device) * (t1 - t0) + t0
        elif dist_options[0] == "logit-normal":
            assert len(dist_options) == 3, "Logit-normal distribution must specify the mean and variance."
            mu, sigma = float(dist_options[1]), float(dist_options[2])
            assert sigma > 0, "Logit-normal distribution must have positive variance."
            t = truncated_logitnormal_sample((bsz,), mu=mu, sigma=sigma, low=t0, high=t1).to(_first_tensor(x1).device)
        else:
            raise NotImplementedError(f"Unknown time distribution type {self.time_dist_type}")

        t = t.to(_first_tensor(x1))
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1

    def _plan_state(self, t, x0, x1):
        if th.is_tensor(x1):
            eff_t, eff_scale = self._component_time(t, self.patch_schedule)
            _, xt, ut = self.patch_schedule.path_sampler.plan(eff_t, x0, x1)
            if self.dual_schedule_mode != "none":
                ut = ut * self._expand_scale_like(eff_scale, ut)
            time_info = EasyDict({
                "base": t,
                "patch": eff_t,
                "patch_scale": eff_scale,
            })
            return t, xt, ut, time_info

        assert isinstance(x1, tuple), f"Unsupported state type: {type(x1)}"
        xt_parts = []
        ut_parts = []
        patch_t = None
        aux_t = None
        aux_scale = None
        for idx, (x0i, x1i) in enumerate(zip(x0, x1)):
            schedule = self._component_schedule(idx)
            eff_t, eff_scale = self._component_time(t, schedule)
            if self._uses_aux_only_mode(x1) and idx == 0:
                xti = x1i
                uti = th.zeros_like(x1i)
            else:
                _, xti, uti = schedule.path_sampler.plan(eff_t, x0i, x1i)
                if self.dual_schedule_mode != "none":
                    uti = uti * self._expand_scale_like(eff_scale, uti)
            xt_parts.append(xti)
            ut_parts.append(uti)
            if idx == 0:
                patch_t = eff_t
            else:
                aux_t = eff_t if aux_t is None else aux_t
                aux_scale = eff_scale if aux_scale is None else aux_scale

        time_info = EasyDict({
            "base": t,
            "patch": patch_t if patch_t is not None else t,
            "aux": aux_t,
            "aux_scale": aux_scale,
        })
        return t, tuple(xt_parts), tuple(ut_parts), time_info

    def _weighted_component_loss(self, model_output, xt, x0, t):
        _, drift_var = self.path_sampler.compute_drift(xt, t)
        sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))

        if self.loss_type in [WeightType.VELOCITY]:
            weight = (drift_var / sigma_t) ** 2
        elif self.loss_type in [WeightType.LIKELIHOOD]:
            weight = drift_var / (sigma_t ** 2)
        elif self.loss_type in [WeightType.NONE]:
            weight = 1
        else:
            raise NotImplementedError()

        if self.model_type == ModelType.NOISE:
            return mean_flat(weight * ((model_output - x0) ** 2))
        else:
            return mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

    def training_losses(self, model, x1, model_kwargs=None, debug_monitor: bool = False):
        """
        Loss for training the model.
        Supports:
            - x1 as tensor
            - x1 as tuple (z, cond)
        """
        if model_kwargs is None:
            model_kwargs = {}

        t, x0, x1 = self.sample(x1)
        t, xt, ut, time_info = self._plan_state(t, x0, x1)
        model_t = self._model_time_input(x1, time_info)
        model_output = model(xt, model_t, **model_kwargs)

        _state_assert_same_shape(model_output, xt)

        terms = {}
        terms["pred"] = model_output

        if self.model_type == ModelType.VELOCITY:
            if th.is_tensor(xt):
                sq_sse, sq_count = _sse_and_count(model_output, ut)
                terms["loss"] = sq_sse / sq_count
            else:
                component_stats = [_sse_and_count(moi, uti) for moi, uti in zip(model_output, ut)]
                patch_sse, patch_n = component_stats[0]
                terms["loss_patch"] = patch_sse / patch_n
                aux_sse = None
                aux_n = 0
                for comp_sse, comp_n in component_stats[1:]:
                    aux_sse = comp_sse if aux_sse is None else aux_sse + comp_sse
                    aux_n += comp_n
                if self._uses_aux_only_mode(x1):
                    patch_zeros = th.zeros_like(terms["loss_patch"])
                    terms["loss_patch"] = patch_zeros
                    if aux_sse is None or aux_n <= 0:
                        raise ValueError("component_mode='aux_only' requires at least one auxiliary component.")
                    aux_loss = aux_sse / aux_n
                    terms["loss_aux"] = aux_loss
                    terms["loss"] = aux_loss
                else:
                    if aux_sse is not None and aux_n > 0:
                        aux_loss = aux_sse / aux_n
                        terms["loss_aux"] = aux_loss
                        terms["loss"] = (patch_sse + self.aux_loss_weight * aux_sse) / (
                            patch_n + self.aux_loss_weight * aux_n
                        )
                    else:
                        terms["loss"] = terms["loss_patch"]
        else:
            if th.is_tensor(xt):
                comp_sse, comp_n = self._weighted_component_sse_and_count(model_output, xt, x0, t)
                terms["loss"] = comp_sse / comp_n
            else:
                component_stats = [
                    self._weighted_component_sse_and_count(moi, xti, x0i, t)
                    for moi, xti, x0i in zip(model_output, xt, x0)
                ]
                patch_sse, patch_n = component_stats[0]
                terms["loss_patch"] = patch_sse / patch_n
                aux_sse = None
                aux_n = 0
                for comp_sse, comp_n in component_stats[1:]:
                    aux_sse = comp_sse if aux_sse is None else aux_sse + comp_sse
                    aux_n += comp_n
                if aux_sse is not None and aux_n > 0:
                    aux_loss = aux_sse / aux_n
                    terms["loss_aux"] = aux_loss
                    terms["loss"] = (patch_sse + self.aux_loss_weight * aux_sse) / (
                        patch_n + self.aux_loss_weight * aux_n
                    )
                else:
                    terms["loss"] = terms["loss_patch"]

        if self._uses_dual_schedules(x1):
            terms["t_base_mean"] = time_info.base.mean()
            terms["t_patch_mean"] = time_info.patch.mean()
            if time_info.aux is not None:
                terms["t_aux_mean"] = time_info.aux.mean()

        if debug_monitor:
            def _tensor_stats(prefix, tensor):
                tensor_f = tensor.detach().float()
                stats = {
                    f"{prefix}_mean": tensor_f.mean(),
                    f"{prefix}_std": tensor_f.std(unbiased=False),
                    f"{prefix}_absmax": tensor_f.abs().max(),
                    f"{prefix}_nonfinite_frac": (~th.isfinite(tensor_f)).float().mean(),
                }
                return stats

            def _state_stats(prefix, state):
                stats = {}
                if th.is_tensor(state):
                    stats.update(_tensor_stats(prefix, state))
                    return stats

                if isinstance(state, tuple):
                    for idx, comp in enumerate(state):
                        label = "patch" if idx == 0 else f"aux{idx}"
                        stats.update(_tensor_stats(f"{prefix}_{label}", comp))
                    return stats

                return stats

            debug_terms = {}
            debug_terms.update(_state_stats("state_xt", xt))
            debug_terms.update(_state_stats("state_ut", ut))
            debug_terms.update(_state_stats("state_pred", model_output))
            terms["debug"] = debug_terms

        return terms

    def get_drift(self):
        """
        Member function for obtaining the drift of the probability flow ODE.
        Supports tensor or tuple state.
        """
        def score_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)

            if th.is_tensor(x):
                drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
                return -drift_mean + drift_var * model_output

            out = []
            for xi, moi in zip(x, model_output):
                drift_mean, drift_var = self.path_sampler.compute_drift(xi, t)
                out.append(-drift_mean + drift_var * moi)
            return tuple(out)

        def noise_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)

            if th.is_tensor(x):
                drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
                sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
                score = model_output / -sigma_t
                return -drift_mean + drift_var * score

            out = []
            for xi, moi in zip(x, model_output):
                drift_mean, drift_var = self.path_sampler.compute_drift(xi, t)
                sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xi))
                score = moi / -sigma_t
                out.append(-drift_mean + drift_var * score)
            return tuple(out)

        def velocity_ode(x, t, model, **model_kwargs):
            return model(x, t, **model_kwargs)

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_t = self._model_time_input(x, self._model_time_from_base(x, t))
            model_output = drift_fn(x, model_t, model, **model_kwargs)
            _state_assert_same_shape(model_output, x)
            if self._uses_aux_only_mode(x):
                if not isinstance(model_output, tuple) or len(model_output) < 2:
                    raise ValueError("aux_only mode expects tuple state/model output")
                zeros = th.zeros_like(x[0])
                return (zeros,) + tuple(model_output[1:])
            return model_output

        return body_fn

    def get_score(self):
        """
        Member function for obtaining score of
            x_t = alpha_t * x + sigma_t * eps
        Supports tensor or tuple state.
        """
        def noise_score(x, t, model, **kwargs):
            model_output = model(x, t, **kwargs)

            if th.is_tensor(x):
                sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
                return model_output / -sigma_t

            out = []
            for xi, moi in zip(x, model_output):
                sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xi))
                out.append(moi / -sigma_t)
            return tuple(out)

        def direct_score(x, t, model, **kwargs):
            return model(x, t, **kwargs)

        def velocity_score(x, t, model, **kwargs):
            model_output = model(x, t, **kwargs)

            if th.is_tensor(x):
                return self.path_sampler.get_score_from_velocity(model_output, x, t)

            out = []
            for xi, moi in zip(x, model_output):
                out.append(self.path_sampler.get_score_from_velocity(moi, xi, t))
            return tuple(out)

        if self.model_type == ModelType.NOISE:
            return noise_score
        elif self.model_type == ModelType.SCORE:
            return direct_score
        elif self.model_type == ModelType.VELOCITY:
            return velocity_score
        else:
            raise NotImplementedError()


class Sampler:
    """Sampler class for the transport model"""

    def __init__(self, transport):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):
        def sde_diffusion_fn(x, t):
            if th.is_tensor(x):
                return self.transport.path_sampler.compute_diffusion(
                    x, t, form=diffusion_form, norm=diffusion_norm
                )

            return tuple(
                self.transport.path_sampler.compute_diffusion(
                    xi, t, form=diffusion_form, norm=diffusion_norm
                )
                for xi in x
            )

        def sde_drift_fn(x, t, model, **kwargs):
            drift_mean = self.drift(x, t, model, **kwargs)
            score = self.score(x, t, model, **kwargs)
            diffusion = sde_diffusion_fn(x, t)
            return _state_map(lambda d, s, g: d - g * s, drift_mean, score, diffusion)

        return sde_drift_fn, sde_diffusion_fn

    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""
        if last_step is None:
            last_step_fn = lambda x, t, model, **model_kwargs: x

        elif last_step == "Mean":
            last_step_fn = lambda x, t, model, **model_kwargs: _state_map(
                lambda xi, di: xi - di * last_step_size,
                x,
                sde_drift(x, t, model, **model_kwargs),
            )

        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t
            sigma = self.transport.path_sampler.compute_sigma_t

            def tweedie_step(x, t, model, **model_kwargs):
                score = self.score(x, t, model, **model_kwargs)

                if th.is_tensor(x):
                    return x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * score

                return tuple(
                    xi / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * si
                    for xi, si in zip(x, score)
                )

            last_step_fn = tweedie_step

        elif last_step == "Euler":
            last_step_fn = lambda x, t, model, **model_kwargs: _state_map(
                lambda xi, di: xi - di * last_step_size,
                x,
                self.drift(x, t, model, **model_kwargs),
            )

        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """
        Returns a sampling function with given SDE settings.
        Supports tensor or tuple state.
        """
        if self.transport.dual_schedule_mode != "none":
            raise NotImplementedError("Dual schedules currently support ODE sampling only.")
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
            time_dist_shift=self.transport.time_dist_shift,
        )

        last_step_fn = self.__get_last_step(
            sde_drift,
            last_step=last_step,
            last_step_size=last_step_size,
        )

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(_batch_size(init), device=_first_tensor(init).device) * (1 - t1)
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"
            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """
        Returns a sampling function with given ODE settings.
        Supports tensor or tuple state.
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            time_dist_shift=self.transport.time_dist_shift,
        )

        return _ode.sample
