import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for folder in ('model_src', 'trainer', 'evaluation'):
    sys.path.insert(0, str(ROOT / folder))

def resolve_config(path):
    config = json.loads(Path(path).read_text())
    def resolve(value):
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if isinstance(value, str):
            value = value.replace('${PACKAGE_ROOT}', str(ROOT))
            value = os.path.expandvars(value)
            if '${' in value:
                raise ValueError('Set the required environment variable for ' + value)
        return value
    return resolve(config)

def configure_register_input(config, register_input):
    import copy
    config = copy.deepcopy(config)
    if register_input is None:
        return config
    if register_input not in ('none', 'native'):
        raise ValueError('Invalid register input')
    params = config['model']['params']
    params['decoder_aux_mode'] = 'prepend' if register_input == 'native' else 'discard'
    params['num_aux_tokens'] = 4 if register_input == 'native' else 0
    config['condition_id'] = config['condition_id'] + '__' + register_input
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--register-input', choices=('none', 'native'))
    args = parser.parse_args()
    from protocol import identity, validate_config, validate_calibration
    from train_decoder import bind_implementation, run_training
    config = validate_config(configure_register_input(resolve_config(args.config), args.register_input))
    binding = bind_implementation(identity(config))
    validate_calibration(config)
    if args.check_inputs:
        print(json.dumps({'status': 'passed', 'binding_sha256': binding['binding_sha256']}))
        return
    try:
        run_training(config, args.output.resolve(), binding, resume=args.resume)
    finally:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    main()
