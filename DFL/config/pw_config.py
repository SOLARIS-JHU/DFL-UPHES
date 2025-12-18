"""
Piecewise (PW) variant configuration.

This configuration uses the piecewise linear approximation MIQP results
as the training data source.
"""

from .base_config import DFLConfig


class PWConfig(DFLConfig):
    """Configuration for Piecewise (PW) variant."""

    def __init__(self):
        """Initialize PW configuration."""
        super().__init__()
        self.variant_name = "PW"
        self.data_file_prefix = "MIQP_piecewise_results"

    def get_miqp_file_path(self):
        """
        Get path to PW MIQP results file.

        Returns:
            str: Path to MIQP_piecewise_results.csv
        """
        return f"{self.miqp_base_path}/MIQP_piecewise/MIQP_piecewise_results.csv"

    def get_data_file_pattern(self, noise_level=None, random_samples=False):
        """
        Get filename pattern for PW training/validation data.

        Args:
            noise_level: Float between 0 and 1 (e.g., 0.1 for 10% noise), or None
            random_samples: Boolean, whether to use random samples dataset

        Returns:
            str: Filename for the PW data file
        """
        if random_samples:
            return f"{self.data_file_prefix}_random_samples.csv"
        elif noise_level is not None:
            noise_pct = int(noise_level * 100)
            return f"{self.data_file_prefix}_relative_noise_{noise_pct:02d}pct.csv"
        else:
            return f"{self.data_file_prefix}.csv"
