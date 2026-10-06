"""Automatic evaluation entry point adapted for the AnisoCast repository."""
import argparse
from pathlib import Path
import subprocess
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parent


def find_latest_checkpoint(model_name, config_file):
    """Find the checkpoint belonging to this configuration, never another run."""
    config_path = Path(config_file).expanduser().resolve()
    with config_path.open() as handle:
        expected = yaml.safe_load(handle)
    directories = [config_path.parent]
    for root in (REPO_ROOT / 'logs', REPO_ROOT / 'lightning_logs'):
        if root.exists():
            for saved_config in root.rglob('config.yaml'):
                with saved_config.open() as handle:
                    saved = yaml.safe_load(handle)
                if saved == expected and saved.get('model_args', {}).get('model_name') == model_name:
                    directories.append(saved_config.parent)
    matches = set()
    for directory in directories:
        checkpoints = directory / 'checkpoints'
        best = checkpoints / 'best.ckpt'
        if best.is_file():
            matches.add(best.resolve())
        elif checkpoints.is_dir():
            matches.update(path.resolve() for path in checkpoints.glob('*.ckpt') if path.name != 'last.ckpt')
    if len(matches) != 1:
        raise ValueError(f'Expected one checkpoint for this configuration, found {len(matches)}. '
                         'Provide --checkpoint_path explicitly.')
    return next(iter(matches))


def auto_evaluate(model_type, config_file, checkpoint_path=None, tag=None, output_dir=None):
    if model_type != 'soon':
        raise ValueError("This repository supports only model_type='soon' (AnisoCast).")
    config_path = Path(config_file).expanduser().resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if config.get('model_args', {}).get('model_name') != 'soon':
        raise ValueError("The configuration must use model_args.model_name='soon'.")
    checkpoint = (Path(checkpoint_path).expanduser().resolve() if checkpoint_path
                  else find_latest_checkpoint(model_type, config_path))
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    final_tag = tag or config.get('runtime_args', {}).get('tag')
    command = [sys.executable, '-m', 'inference.evaluate_soon',
               '--config_filepath', str(config_path), '--checkpoint_path', str(checkpoint)]
    if final_tag:
        command.extend(['--tag', str(final_tag)])
    if output_dir:
        command.extend(['--output_dir', str(Path(output_dir).expanduser().resolve())])
    print(f'Evaluating AnisoCast checkpoint: {checkpoint}', flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_type', required=True, choices=['soon'], help='AnisoCast legacy model identifier')
    parser.add_argument('--config_file', required=True, help='Training configuration YAML')
    parser.add_argument('--checkpoint_path', help='Checkpoint; required if automatic selection is ambiguous')
    parser.add_argument('--tag', help='Optional suffix for the metrics CSV')
    parser.add_argument('--output_dir', help='Directory for the evaluation results')
    args = parser.parse_args()
    try:
        success = auto_evaluate(args.model_type, args.config_file, args.checkpoint_path,
                                args.tag, args.output_dir)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f'Evaluation failed: {error}', file=sys.stderr)
        return 1
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
