from .transport import Transport, ModelType, WeightType, PathType, Sampler


def _resolve_path_type(path_type):
    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }
    if isinstance(path_type, PathType):
        return path_type
    return path_choice[path_type]


def _default_eps(path_type, model_type):
    if path_type in [PathType.VP]:
        return 1e-5, 1e-3
    if path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        return 1e-3, 1e-3
    return 0, 0


def _resolve_branch_schedule(schedule_cfg, *, base_path_type, base_train_eps, base_sample_eps, base_time_shift):
    cfg = {} if schedule_cfg is None else dict(schedule_cfg)
    raw_path_type = cfg.get("path_type", "inherit")
    if raw_path_type in (None, "inherit"):
        path_type = base_path_type
    else:
        path_type = _resolve_path_type(raw_path_type)

    raw_train_eps = cfg.get("train_eps", "inherit")
    raw_sample_eps = cfg.get("sample_eps", "inherit")
    train_eps = base_train_eps if raw_train_eps in (None, "inherit") else float(raw_train_eps)
    sample_eps = base_sample_eps if raw_sample_eps in (None, "inherit") else float(raw_sample_eps)

    raw_time_shift = cfg.get("time_shift", "inherit")
    time_shift = base_time_shift if raw_time_shift in (None, "inherit") else float(raw_time_shift)

    return {
        "path_type": path_type,
        "train_eps": train_eps,
        "sample_eps": sample_eps,
        "time_shift": time_shift,
    }


def create_transport(
    path_type='Linear',
    prediction="velocity",
    loss_weight=None,
    aux_loss_weight=1.0,
    component_mode="joint",
    train_eps=None,
    sample_eps=None,
    time_dist_type="uniform",
    time_dist_shift=1.0,
    dual_schedule_mode="none",
    patch_schedule=None,
    aux_schedule=None,
):
    """function for creating Transport object
    **Note**: model prediction defaults to velocity
    Args:
    - path_type: type of path to use; default to linear
    - learn_score: set model prediction to score
    - learn_noise: set model prediction to noise
    - velocity_weighted: weight loss by velocity weight
    - likelihood_weighted: weight loss by likelihood weight
    - train_eps: small epsilon for avoiding instability during training
    - sample_eps: small epsilon for avoiding instability during sampling
    - time_dist_type: type of time distribution to use; default to uniform
    - time_dist_shift: shift for time distribution; default to 1.0
    """

    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_type = _resolve_path_type(path_type)

    default_train_eps, default_sample_eps = _default_eps(path_type, model_type)
    train_eps = default_train_eps if train_eps is None else float(train_eps)
    sample_eps = default_sample_eps if sample_eps is None else float(sample_eps)

    patch_schedule = _resolve_branch_schedule(
        patch_schedule,
        base_path_type=path_type,
        base_train_eps=train_eps,
        base_sample_eps=sample_eps,
        base_time_shift=float(time_dist_shift),
    )
    aux_schedule = _resolve_branch_schedule(
        aux_schedule,
        base_path_type=path_type,
        base_train_eps=train_eps,
        base_sample_eps=sample_eps,
        base_time_shift=float(time_dist_shift),
    )
    
    # create flow state
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        aux_loss_weight=aux_loss_weight,
        component_mode=component_mode,
        time_dist_type=time_dist_type,
        time_dist_shift=time_dist_shift,
        train_eps=train_eps,
        sample_eps=sample_eps,
        dual_schedule_mode=dual_schedule_mode,
        patch_schedule=patch_schedule,
        aux_schedule=aux_schedule,
    )
    
    return state
