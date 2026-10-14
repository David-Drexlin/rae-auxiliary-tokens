from contextlib import contextmanager
from copy import deepcopy
import re
import threading
import warnings

import torch


POLICY = {
    'schema': 'register-preservation-backward-policy-v1',
    'allowed_operator': 'upsample_bicubic2d_backward_out_cuda',
    'scope': 'register_preservation CUDA autograd execution only',
    'deterministic_algorithms': True,
    'unexpected_determinism_warning': 'raise before optimizer step',
    'forward_preprocessing': 'unchanged',
    'restoration_and_input_checks': 'exact; no tolerance allowance',
    'trajectory_admission': 'separate prospective numerical validation required',
}
_ALLOWED = re.compile(
    r'^upsample_bicubic2d_backward_out_cuda does not have a deterministic implementation\b'
)
_DETERMINISM = re.compile(r'non[- ]?deterministic|determinism|deterministic', re.I)
_LOCK = threading.RLock()
_LOCAL = threading.local()
_EVIDENCE = {'guard_calls': {}, 'allowed_warning_count': 0,
             'allowed_warnings': {}, 'unexpected_warnings': []}


class UnexpectedNondeterminism(RuntimeError):
    pass


def runtime_evidence():

    with _LOCK:
        return {'policy': deepcopy(POLICY), **deepcopy(_EVIDENCE)}


@contextmanager
def preservation_backward(model, device, *, phase):






    objective = getattr(model, 'objective', model)
    enabled = (getattr(objective, 'kind', None) == 'register_preservation'
               and torch.device(device).type == 'cuda')
    if not enabled:
        yield
        return
    if not isinstance(phase, str) or not phase:
        raise ValueError('Preservation backward requires a named execution phase')
    with _LOCK:
        stack = getattr(_LOCAL, 'phases', [])
        _EVIDENCE['guard_calls'][phase] = _EVIDENCE['guard_calls'].get(phase, 0) + 1
        if stack:
            stack.append(phase)
            try:
                yield
            finally:
                stack.pop()
            return
        if (not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()):
            raise RuntimeError('Preservation allowance requires strict deterministic mode outside backward')
        prior_warn_always = torch.is_warn_always_enabled()
        prior_showwarning = warnings.showwarning
        stack = [phase]
        _LOCAL.phases = stack

        def checked_warning(message, category, filename, lineno, file=None, line=None):
            text = str(message)
            event = {'phase': stack[-1], 'message': text,
                     'category': category.__name__}
            if _ALLOWED.match(text):
                _EVIDENCE['allowed_warning_count'] += 1
                key = stack[-1]
                if key not in _EVIDENCE['allowed_warnings']:
                    _EVIDENCE['allowed_warnings'][key] = {**event, 'count': 0}
                _EVIDENCE['allowed_warnings'][key]['count'] += 1
                return
            if _DETERMINISM.search(text):
                _EVIDENCE['unexpected_warnings'].append(event)
                raise UnexpectedNondeterminism(
                    'Unapproved nondeterminism in '+stack[-1]+': '+text)
            prior_showwarning(message, category, filename, lineno, file=file, line=line)

        try:
            with warnings.catch_warnings():



                warnings.simplefilter('always')
                warnings.showwarning = checked_warning
                torch.set_warn_always(True)
                torch.use_deterministic_algorithms(True, warn_only=True)
                try:
                    yield
                finally:
                    torch.use_deterministic_algorithms(True, warn_only=False)
                    torch.set_warn_always(prior_warn_always)
        finally:
            _LOCAL.phases = []
