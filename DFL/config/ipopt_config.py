"""
IPOPT NLP comparison configuration.

These configurations use MIQP solutions as warm-starts for IPOPT NLP solving.
The NLP uses true nonlinear constraints (no Taylor approximation) to benchmark
the quality of DFL's linearization approach.
"""

from .base_config import DFLConfig


class IPOPTConfigPW(DFLConfig):
    """Configuration for IPOPT NLP with Piecewise MIQP warm-starts."""

    def __init__(self):
        """Initialize IPOPT PW configuration."""
        super().__init__()
        self.variant_name = "IPOPT-NLP"
        self.data_file_prefix = "MIQP_piecewise_results"
        self.data_file_pattern = "MIQP_piecewise_results_relative_noise_{noise}.csv"
        self.random_samples_file = "MIQP_piecewise_results_random_samples.csv"

        # No neural network
        self.use_neural_network = False

        # IPOPT solver settings
        self.ipopt_max_iter = 10000
        self.ipopt_tol = 1e-6
        self.ipopt_time_limit = 3600

    def get_miqp_file_path(self):
        """Get path to PW MIQP results file."""
        return f"{self.miqp_base_path}/MIQP_piecewise/MIQP_piecewise_results.csv"

    def get_data_file_pattern(self, noise_level=None, random_samples=False):
        """Get filename pattern for data files."""
        base_dir = self.data_dir
        if random_samples:
            return f"{base_dir}/{self.data_file_prefix}_random_samples.csv"
        elif noise_level is not None:
            noise_pct = int(noise_level * 100)
            return f"{base_dir}/{self.data_file_prefix}_relative_noise_{noise_pct:02d}pct.csv"
        else:
            return f"{base_dir}/{self.data_file_prefix}.csv"


class IPOPTConfigGL(DFLConfig):
    """Configuration for IPOPT NLP with Global Linear MIQP warm-starts."""

    def __init__(self):
        """Initialize IPOPT GL configuration."""
        super().__init__()
        self.variant_name = "IPOPT-NLP"
        self.data_file_prefix = "MIQP_linear_results"
        self.data_file_pattern = "MIQP_linear_results_relative_noise_{noise}.csv"
        self.random_samples_file = "MIQP_linear_results_random_samples.csv"

        # No neural network
        self.use_neural_network = False

        # IPOPT solver settings
        self.ipopt_max_iter = 10000
        self.ipopt_tol = 1e-6
        self.ipopt_time_limit = 3600

    def get_miqp_file_path(self):
        """Get path to GL MIQP results file."""
        return f"{self.miqp_base_path}/MIQP_linear/MILP_global_linear_results.csv"

    def get_data_file_pattern(self, noise_level=None, random_samples=False):
        """Get filename pattern for data files."""
        base_dir = self.data_dir
        if random_samples:
            return f"{base_dir}/{self.data_file_prefix}_random_samples.csv"
        elif noise_level is not None:
            noise_pct = int(noise_level * 100)
            return f"{base_dir}/{self.data_file_prefix}_relative_noise_{noise_pct:02d}pct.csv"
        else:
            return f"{base_dir}/{self.data_file_prefix}.csv"
