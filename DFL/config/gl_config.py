"""
Global Linear (GL) variant configuration.

This configuration uses the global linear approximation MIQP results
as the training data source.
"""

from .base_config import DFLConfig


class GLConfig(DFLConfig):
    """Configuration for Global Linear (GL) variant."""

    def __init__(self):
        """Initialize GL configuration."""
        super().__init__()
        self.variant_name = "GL"
        self.data_file_prefix = "MIQP_linear_results"
        # File pattern with {noise} placeholder for substitution
        self.data_file_pattern = "MIQP_linear_results_relative_noise_{noise}.csv"
        self.random_samples_file = "MIQP_linear_results_random_samples.csv"

    def get_miqp_file_path(self):
        """
        Get path to GL MIQP results file.

        Returns:
            str: Path to MILP_global_linear_results.csv
        """
        return f"{self.miqp_base_path}/MIQP_linear/MILP_global_linear_results.csv"

    def get_data_file_pattern(self, noise_level=None, random_samples=False):
        """
        Get filename pattern for GL training/validation data.

        Args:
            noise_level: Float between 0 and 1 (e.g., 0.1 for 10% noise), or None
            random_samples: Boolean, whether to use random samples dataset

        Returns:
            str: Filename for the GL data file
        """
        if random_samples:
            return f"{self.data_file_prefix}_random_samples.csv"
        elif noise_level is not None:
            noise_pct = int(noise_level * 100)
            return f"{self.data_file_prefix}_relative_noise_{noise_pct:02d}pct.csv"
        else:
            return f"{self.data_file_prefix}.csv"
