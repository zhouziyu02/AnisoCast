"""Synthetic runtime checks; no ERA5 training or scientific skill reproduction."""
from pathlib import Path
import tempfile
import unittest

import lightning.pytorch as pl
import numpy as np
import torch
import xarray as xr

from CIRT import config, criterion
from CIRT.dataset import S2SDataset
from CIRT.models.model import S2SBenchmarkModel
from CIRT.models.soon import Model
from inference.evaluate_soon import load_model_and_predict, resolve_hyperparameters, reverse_normalize


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.single = ['2m_temperature']
        cls.pressure = list(config.ERA5_PRESSURE_LIST)
        cls.params = [f'{var}-{level}' for var in cls.pressure for level in config.PRESSURE_LEVELS]
        for dirname in ('pressure_level_1.5', 'single_level_1.5', 'climatology_1.5'):
            (cls.root / dirname).mkdir()
        coords = {'level': config.PRESSURE_LEVELS, 'latitude': np.linspace(90, -90, 4),
                  'longitude': np.arange(8)}
        base = np.arange(60, dtype=np.float32).reshape(6, 10) * 100
        for day in range(45):
            date = (np.datetime64('2000-01-01') + np.timedelta64(day, 'D')).astype('datetime64[ns]')
            ymd = str(date.astype('datetime64[D]')).replace('-', '')
            # Store noncanonical dimension order to verify explicit axis selection.
            fields = {var: (('longitude', 'latitude', 'level'),
                            np.broadcast_to(base[index][None, None, :] + day, (8, 4, 10)).copy())
                      for index, var in enumerate(cls.pressure)}
            xr.Dataset(fields, coords={**coords, 'time': date}).to_zarr(
                cls.root / 'pressure_level_1.5' / f'pressure_{ymd}.zarr', consolidated=True)
            xr.Dataset({'2m_temperature': (('latitude', 'longitude'),
                                          np.full((4, 8), 1000 + day, dtype=np.float32))},
                       coords={'latitude': coords['latitude'], 'longitude': coords['longitude'], 'time': date}).to_zarr(
                cls.root / 'single_level_1.5' / f'single_{ymd}.zarr', consolidated=True)
        # Reverse the statistics order to expose positional normalization errors.
        xr.Dataset({'mean': ('param', np.arange(60, dtype=np.float32)[::-1] * 100),
                    'sigma': ('param', np.full(60, 2, dtype=np.float32))},
                   coords={'param': cls.params[::-1]}).to_zarr(
            cls.root / 'climatology_1.5' / 'climatology_pressure_level_1.5_new.zarr', consolidated=True)
        xr.Dataset({'mean': ('param', [1000.0]), 'sigma': ('param', [4.0])},
                   coords={'param': cls.single}).to_zarr(
            cls.root / 'climatology_1.5' / 'climatology_single_level_1.5_new.zarr', consolidated=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def make_dataset(self, **kwargs):
        args = dict(data_dir=str(self.root), years=[2000], n_step=28, lead_time=15,
                    single_vars=self.single, pred_single_vars=self.single,
                    pred_pressure_vars=self.pressure)
        args.update(kwargs)
        return S2SDataset(**args)

    def test_windows_last_sample_and_label_order_roundtrip(self):
        data = self.make_dataset()
        self.assertEqual(len(data), 3)
        timestamp, x, y = data[0]
        self.assertEqual(tuple(x.shape), (61, 4, 8))
        self.assertEqual(tuple(y.shape), (2, 61, 4, 8))
        self.assertEqual(timestamp, int(np.datetime64('2000-01-01', 'ns').astype(np.int64)))
        torch.testing.assert_close(x, torch.zeros_like(x))
        torch.testing.assert_close(y[0, :60], torch.full((60, 4, 8), 21.5 / 2))
        torch.testing.assert_close(y[1, :60], torch.full((60, 4, 8), 35.5 / 2))
        torch.testing.assert_close(data[2][2][1, 0], torch.full((4, 8), 37.5 / 2))
        with self.assertRaises(IndexError):
            data[3]
        raw = reverse_normalize(y.unsqueeze(0), {
            'data_dir': str(self.root), 'pred_pressure_vars': self.pressure,
            'pred_single_vars': self.single,
        })
        torch.testing.assert_close(raw[0, 0, :60, 0, 0], torch.arange(60) * 100.0 + 21.5)
        self.assertEqual(raw[0, 1, 60, 0, 0].item(), 1035.5)

    def test_rejects_unpaired_dates_missing_year_and_invalid_window(self):
        source = self.root / 'single_level_1.5' / 'single_20000120.zarr'
        hidden = source.with_suffix('.hidden')
        source.rename(hidden)
        try:
            with self.assertRaisesRegex(ValueError, 'dates do not match'):
                self.make_dataset()
        finally:
            hidden.rename(source)
        with self.assertRaises(FileNotFoundError):
            self.make_dataset(years=[1999])
        with self.assertRaises(ValueError):
            self.make_dataset(n_step=1)

    def test_rejects_missing_day_in_both_sources(self):
        sources = [self.root / 'pressure_level_1.5' / 'pressure_20000120.zarr',
                   self.root / 'single_level_1.5' / 'single_20000120.zarr']
        for source in sources:
            source.rename(source.with_suffix('.hidden'))
        try:
            with self.assertRaisesRegex(ValueError, 'not contiguous'):
                self.make_dataset()
        finally:
            for source in sources:
                source.with_suffix('.hidden').rename(source)

    def test_acc_matches_single_cosine_weight_with_missing_cells(self):
        metric = criterion.ACC(data_dir=self.root)
        torch.manual_seed(7)
        pred, target = torch.randn(2, 121, 3), torch.randn(2, 121, 3)
        pred[0, 50, 0] = float('nan')
        valid = torch.isfinite(pred) & torch.isfinite(target)
        p, t = torch.where(valid, pred, 0), torch.where(valid, target, 0)
        weights = criterion.get_adjusting_weights().clamp_min(0)
        expected = (weights * p * t).sum() / torch.sqrt(
            (weights * p.square()).sum() * (weights * t.square()).sum())
        actual = metric(pred, target, self.params[0], 'pressure_level')
        self.assertEqual(actual.ndim, 0)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(metric(target, target, self.params[0], 'pressure_level'), torch.tensor(1.0))
        self.assertTrue(torch.isnan(metric(torch.zeros_like(target), torch.zeros_like(target),
                                          self.params[0], 'pressure_level')))

    def test_model_forward_backward_preserves_output_shape_and_mse(self):
        network = Model(img_size=(4, 8), input_size=3, embed_dim=8, depth=2,
                        decoder_depth=1, drop_path=0, drop_rate=0)
        x = torch.randn(2, 3, 4, 8)
        pred = network(x)
        self.assertEqual(tuple(pred.shape), (2, 2, 3, 4, 8))
        target = torch.randn_like(pred)
        loss = criterion.MSE()(pred, target)
        torch.testing.assert_close(loss, (pred - target).square().mean())
        loss.backward()
        self.assertTrue(all(param.grad is None or torch.isfinite(param.grad).all()
                            for param in network.parameters()))

    def test_ms_ssim_constant_fields_are_finite(self):
        fields = torch.full((2, 121, 16), 7.0)
        metric = criterion.MS_SSIM()
        value = metric(fields, fields)
        self.assertTrue(torch.isfinite(value))
        torch.testing.assert_close(value, torch.tensor(1.0))
        varied = torch.randn_like(fields)
        torch.testing.assert_close(metric(varied, varied), torch.tensor(1.0))

    def test_saved_architecture_data_override_strict_load_and_tail_batch(self):
        model_args = dict(model_name='soon', input_size=61, output_size=61, pred_len=2,
                          img_size=[4, 8], embed_dim=8, depth=1, decoder_depth=1,
                          drop_path=0, drop_rate=0, num_workers=0)
        data_args = dict(data_dir=str(self.root), test_years=[2000], batch_size=2,
                         n_step=28, lead_time=15, single_vars=self.single,
                         pred_single_vars=self.single, pred_pressure_vars=self.pressure)
        network = S2SBenchmarkModel(model_args, data_args)
        checkpoint = {'state_dict': network.state_dict(),
                      'hyper_parameters': {'model_args': model_args, 'data_args': data_args},
                      'pytorch-lightning_version': pl.__version__}
        resolved_model, resolved_data = resolve_hyperparameters(checkpoint, {
            'model_args': {'embed_dim': 99, 'num_workers': 0},
            'data_args': {'data_dir': str(self.root), 'test_years': [2000], 'lead_time': 1},
        })
        self.assertEqual(resolved_model['embed_dim'], 8)
        self.assertEqual(resolved_data['lead_time'], 15)
        self.assertEqual(resolved_data['data_dir'], str(self.root.resolve()))
        checkpoint_path = self.root / 'synthetic.ckpt'
        torch.save(checkpoint, checkpoint_path)
        pred, target = load_model_and_predict(resolved_model, resolved_data, checkpoint_path)
        self.assertEqual(tuple(pred.shape), (3, 2, 61, 4, 8))
        self.assertEqual(pred.shape, target.shape)
        self.assertTrue(torch.isfinite(pred).all())
        torch.testing.assert_close(target[:, 0, 0, 0, 0], torch.tensor([21.5, 22.5, 23.5]))
        checkpoint['state_dict'] = dict(checkpoint['state_dict'])
        checkpoint['state_dict'].pop('model.pos_embed')
        torch.save(checkpoint, checkpoint_path)
        with self.assertRaises(RuntimeError):
            load_model_and_predict(resolved_model, resolved_data, checkpoint_path)


if __name__ == '__main__':
    unittest.main()
