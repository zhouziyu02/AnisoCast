"""Small offline regressions for ERA5 preprocessing fixes."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

SCRIPTS = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(SCRIPTS))
import calculate_climatology as climatology
import parallel_download_pressure_1p5 as pressure
import parallel_download_single_1p5 as single


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def dataset(self, values, times, levels=None):
        dimensions = ("time", "level", "latitude", "longitude") if levels is not None else ("time", "latitude", "longitude")
        coordinates = {"time": pd.to_datetime(times), "latitude": np.arange(values.shape[-2]),
                       "longitude": np.arange(values.shape[-1])}
        if levels is not None:
            coordinates["level"] = levels
        return xr.Dataset({"temperature": (dimensions, values)}, coords=coordinates)

    def write(self, name, dataset):
        path = self.root / name
        dataset.to_zarr(path, consolidated=True)
        return path

    def test_multitime_stable_variance_and_raw_reader(self):
        first = 1e8 + np.arange(16, dtype="float64").reshape(2, 2, 2, 2)
        second = 1e8 + np.arange(16, 24, dtype="float64").reshape(1, 2, 2, 2)
        first[0, 0, 0, 0] = np.nan
        paths = [self.write("first.zarr", self.dataset(first, ["2016-12-30T00:00", "2016-12-30T12:00"], [10, 50])),
                 self.write("second.zarr", self.dataset(second, ["2016-12-31"], [10, 50]))]
        expected = np.concatenate([first, second])
        for aggregation in ("pairwise", "concat"):
            for raw in (False, True):
                with redirect_stdout(io.StringIO()):
                    output = climatology.compute_climatology(paths, ["temperature"], True, agg_mode=aggregation,
                                                            force_raw_zarr=raw, progress_mode="none",
                                                            start="2016-12-30", end="2016-12-31")
                np.testing.assert_allclose(output["mean"], np.nanmean(expected, axis=(0, 2, 3)), rtol=0, atol=1e-7)
                np.testing.assert_allclose(output["sigma"], np.nanstd(expected, axis=(0, 2, 3)), rtol=1e-8)

    def test_training_date_filter_and_missing_day_failure(self):
        train = self.write("train.zarr", self.dataset(np.ones((1, 2, 2)), ["2016-12-31"]))
        validation = self.write("validation.zarr", self.dataset(np.full((1, 2, 2), 999.), ["2017-01-01"]))
        with redirect_stdout(io.StringIO()):
            result = climatology.compute_climatology([train, validation], ["temperature"], False,
                                                    progress_mode="none", start="2016-12-31", end="2016-12-31")
        np.testing.assert_allclose(result["mean"], [1.])
        self.assertEqual(result.attrs["period_end"], "2016-12-31")
        with self.assertRaisesRegex(ValueError, "requested day.*missing"):
            climatology.compute_climatology([train], ["temperature"], False, progress_mode="none",
                                            start="2016-12-30", end="2016-12-31")

    def test_no_overwrite_preserves_existing_outputs_for_both_downloaders(self):
        source = self.dataset(np.arange(8, dtype="float32").reshape(2, 2, 2),
                              ["2016-12-31T00:00", "2016-12-31T12:00"])
        single.GLOBAL_DS = source
        pressure.GLOBAL_DS = source.expand_dims(level=[10], axis=1)
        date = pd.Timestamp("2016-12-31")
        with redirect_stdout(io.StringIO()):
            path = single.process_one_day(date, str(self.root), ["temperature"], 1, 1, True, "sync", 1)
            single.GLOBAL_DS = source + 1000
            single.process_one_day(date, str(self.root), ["temperature"], 1, 1, False, "sync", 1)
            pressure_path = pressure.process_one_day(date, str(self.root), ["temperature"], [10], 1, 1, True)
            pressure.GLOBAL_DS = pressure.GLOBAL_DS + 1000
            pressure.process_one_day(date, str(self.root), ["temperature"], [10], 1, 1, False)
        for output_path in (path, pressure_path):
            with xr.open_zarr(output_path) as output:
                self.assertEqual(float(output["temperature"].max()), 7.)
                self.assertEqual(output.sizes["time"], 2)

    def test_download_failures_return_nonzero(self):
        source = self.dataset(np.ones((1, 2, 2)), ["2016-12-31"])
        for module in (single, pressure):
            arguments = ["download", "--start", "2016-12-30", "--end", "2016-12-30", "--workers", "1",
                         "--retries", "1", "--output_dir", str(self.root)]
            with patch.object(module.xr, "open_zarr", return_value=source), patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), 1)

    def test_shell_wrappers_propagate_failure(self):
        executable = self.root / "failed-python"
        executable.write_text("#!/bin/sh\nexit 7\n")
        executable.chmod(0o755)
        environment = dict(os.environ, PYTHON=str(executable), BATCH_JOBS="2", WORKERS="1", LOG_DIR=str(self.root / "logs"))
        for name in ("download_single.sh", "download_pressure.sh"):
            completed = subprocess.run(["bash", str(SCRIPTS / name)], env=environment, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1, completed.stderr)


if __name__ == "__main__":
    unittest.main()
