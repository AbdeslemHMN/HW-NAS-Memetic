import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.hw_nas_wrapper import HWNASApi


class TestHWNASDataLoading(unittest.TestCase):
    def setUp(self):
        self.data_dir = PROJECT_ROOT / "data"
        self.pickle_path = self.data_dir / "HW-NAS-Bench-v1_0.pickle"

    def test_data_directory_exists(self):
        """The project must contain a top-level data/ directory."""
        self.assertTrue(
            self.data_dir.is_dir(),
            f"Expected data directory to exist at: {self.data_dir}"
        )

    def test_hwnas_pickle_is_available(self):
        """HW-NAS-Bench pickle file must be present in data/."""
        self.assertTrue(
            self.pickle_path.is_file(),
            f"Expected HW-NAS-Bench pickle file to exist at: {self.pickle_path}"
        )

    def test_hwnas_api_loads_benchmark(self):
        """The HWNASApi should successfully load the pickle and provide at least one device."""
        api = HWNASApi(str(self.pickle_path))
        self.assertTrue(api.devices, "Expected at least one hardware device in the benchmark.")
        self.assertIsInstance(api.devices[0], str)

    def test_hwnas_api_query_returns_metrics(self):
        """A sample architecture query should return a valid latency metric."""
        api = HWNASApi(str(self.pickle_path))
        config0 = api.data['nasbench201']['cifar10']['config'][0]
        sample_arch = api._arch_str_to_tuple(config0['arch_str'])

        latency = api.query(sample_arch, device='nasbench201')
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 0.0)

    def test_hwnas_api_query_all_datasets_and_metrics(self):
        """All NAS-Bench-201 datasets and hardware metrics must be queryable."""
        api = HWNASApi(str(self.pickle_path))
        datasets = [
            key for key in api.data['nasbench201'].keys()
            if key != 'config'
        ]
        self.assertGreater(len(datasets), 0, "Expected at least one dataset split.")

        sample_config = api.data['nasbench201']['cifar10']['config'][0]
        sample_arch = api._arch_str_to_tuple(sample_config['arch_str'])

        metrics = [
            key for key in api.data['nasbench201']['cifar10'].keys()
            if key != 'config'
        ]
        self.assertGreater(len(metrics), 0, "Expected at least one hardware metric.")

        for dataset in datasets:
            accuracy = api.query_accuracy(sample_arch, dataset=dataset)
            self.assertIsInstance(accuracy, float)
            self.assertGreaterEqual(accuracy, 0.0)

            for metric in metrics:
                latency = api.query(
                    sample_arch,
                    device='nasbench201',
                    dataset=dataset,
                    metric=metric,
                )
                self.assertIsInstance(latency, float)
                self.assertGreaterEqual(latency, 0.0)

    def test_query_too_short_vector_raises_value_error(self):
        """A vector shorter than 6 elements must raise ValueError."""
        api = HWNASApi(str(self.pickle_path))
        with self.assertRaises(ValueError):
            api.query([0, 1, 2], device='nasbench201')

    def test_query_too_long_vector_raises_value_error(self):
        """A vector longer than 6 elements must raise ValueError."""
        api = HWNASApi(str(self.pickle_path))
        with self.assertRaises(ValueError):
            api.query([0, 1, 2, 3, 4, 0, 1], device='nasbench201')

    def test_query_unknown_device_raises_value_error(self):
        """An unrecognised device name must raise ValueError."""
        api = HWNASApi(str(self.pickle_path))
        config0 = api.data['nasbench201']['cifar10']['config'][0]
        valid_arch = api._arch_str_to_tuple(config0['arch_str'])
        with self.assertRaises(ValueError):
            api.query(valid_arch, device='nonexistent_device')

    def test_query_unknown_architecture_raises_key_error(self):
        """An architecture tuple not present in the benchmark must raise KeyError."""
        api = HWNASApi(str(self.pickle_path))
        # (5,5,5,5,5,5) is out of range for ops 0-4, guaranteed not in index
        with self.assertRaises((KeyError, ValueError)):
            api.query([5, 5, 5, 5, 5, 5], device='nasbench201')

    def test_query_unknown_metric_raises_key_error(self):
        """Querying with a non-existent metric name must raise KeyError."""
        api = HWNASApi(str(self.pickle_path))
        config0 = api.data['nasbench201']['cifar10']['config'][0]
        valid_arch = api._arch_str_to_tuple(config0['arch_str'])
        with self.assertRaises(KeyError):
            api.query(valid_arch, device='nasbench201', metric='nonexistent_metric')


if __name__ == "__main__":
    unittest.main()
