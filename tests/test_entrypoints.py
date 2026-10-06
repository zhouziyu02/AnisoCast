"""Regression checks for evaluator routing and failure propagation."""
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import yaml

import auto_evaluate


@pytest.fixture
def evaluation_run(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'model_args': {'model_name': 'soon'}, 'runtime_args': {'tag': 'sample'}}))
    (tmp_path / 'checkpoints').mkdir()
    checkpoint = tmp_path / 'checkpoints' / 'best.ckpt'
    checkpoint.write_bytes(b'Routing fixture, not model weights')
    return config, checkpoint


def test_auto_evaluator_forwards_supported_entry_and_propagates_failure(evaluation_run, tmp_path):
    config, checkpoint = evaluation_run
    for status, expected in [(0, True), (7, False)]:
        with patch('auto_evaluate.subprocess.run', return_value=subprocess.CompletedProcess([], status)) as run:
            assert auto_evaluate.auto_evaluate('soon', config, checkpoint, output_dir=tmp_path / 'scores') is expected
        command = run.call_args.args[0]
        assert command[:3] == [sys.executable, '-m', 'inference.evaluate_soon']
        assert command[command.index('--checkpoint_path') + 1] == str(checkpoint)
        assert command[command.index('--tag') + 1] == 'sample'
        assert run.call_args.kwargs['cwd'] == auto_evaluate.REPO_ROOT


def test_checkpoint_discovery_does_not_select_an_unrelated_run(evaluation_run, tmp_path):
    config, checkpoint = evaluation_run
    with patch('auto_evaluate.REPO_ROOT', tmp_path):
        assert auto_evaluate.find_latest_checkpoint('soon', config) == checkpoint
        checkpoint.unlink()
        unrelated = tmp_path / 'logs' / 'another'
        (unrelated / 'checkpoints').mkdir(parents=True)
        (unrelated / 'config.yaml').write_text(yaml.safe_dump({'model_args': {'model_name': 'different'}}))
        (unrelated / 'checkpoints' / 'best.ckpt').write_bytes(b'Unrelated fixture')
        with pytest.raises(ValueError, match='found 0'):
            auto_evaluate.find_latest_checkpoint('soon', config)


def test_auto_evaluator_rejects_missing_checkpoint(evaluation_run):
    config, checkpoint = evaluation_run
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError):
        auto_evaluate.auto_evaluate('soon', config, checkpoint)


def test_training_launcher_preserves_failed_process_status(tmp_path):
    stub = tmp_path / 'python-stub'
    stub.write_text('#!/usr/bin/env bash\nexit 7\n')
    stub.chmod(0o755)
    import os
    env = {**os.environ, 'PYTHON': str(stub)}
    env.pop('CUDA_VISIBLE_DEVICES', None)
    result = subprocess.run(
        ['bash', str(auto_evaluate.REPO_ROOT / 'train_ours.sh'), '--np', '1',
         '--foreground', '--tag', 'failed-run', '--log-dir', str(tmp_path / 'logs')],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 7, result.stdout + result.stderr
    assert 'Training failed with exit code 7' in result.stdout
