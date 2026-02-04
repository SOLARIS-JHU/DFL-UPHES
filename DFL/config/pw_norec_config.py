"""
Piecewise (PW) No-Recursion variant configuration.

This configuration uses the piecewise linear approximation MIQP results
as the training data source with only a single iteration (no recursive refinement)
but with the neural network enabled for weight prediction.
"""

from .pw_config import PWConfig


class PWNoRecConfig(PWConfig):
    """
    Configuration for Piecewise No-Recursion (PW-no-Rec) variant.

    Inherits from PWConfig (uses same data source: piecewise MIQP),
    but limits to single iteration with neural network predictions.
    This variant tests the impact of recursive refinement.
    """

    def __init__(self):
        """Initialize PW-no-Rec configuration."""
        super().__init__()
        self.variant_name = "PW-no-Rec"

        self.max_iterations = 1
        self.init_w_p = 3.0
        self.init_w_q = 0.2
        self.init_w_h = 5.0
        self.w_p_min = 0.6
        self.w_q_min = 0.02
        self.w_h_min = 0.1
        self.w_p_max = 3.0
        self.w_q_max = 0.2
        self.w_h_max = 5.0

    def get_data_file_pattern(self, noise_level=None, random_samples=False):
        """
        Get filename pattern for PW-no-Rec training/validation data.

        Args:
            noise_level: Float between 0 and 1 (e.g., 0.1 for 10% noise), or None
            random_samples: Boolean, whether to use random samples dataset

        Returns:
            str: Full path to the PW-no-Rec data file
        """
        base_dir = self.data_dir
        if random_samples:
            return f"{base_dir}/{self.data_file_prefix}_random_samples.csv"
        elif noise_level is not None:
            noise_pct = int(noise_level * 100)
            return f"{base_dir}/{self.data_file_prefix}_relative_noise_{noise_pct:02d}pct.csv"
        else:
            return f"{base_dir}/{self.data_file_prefix}.csv"
