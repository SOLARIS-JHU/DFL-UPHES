"""
Piecewise Fixed-Weight variant configuration (No Neural Network).

This configuration tests the recursive linearization approach without
the neural network component, using fixed penalty weights instead.
Used as a baseline to measure the impact of the neural network predictor.
"""

from .pw_config import PWConfig


class AblationConfig(PWConfig):
    """
    Configuration for Piecewise Fixed-Weight variant (No Neural Network).

    Inherits from PWConfig since it uses the same data source (piecewise MIQP),
    but disables the neural network and uses fixed weights instead of learned weights.
    Uses 7 iterations for recursive refinement with fixed penalty weights.
    """

    def __init__(self):
        """Initialize Ablation configuration."""
        super().__init__()
        self.variant_name = "Ablation"

        # Key difference: disable neural network
        self.use_neural_network = False

        # Fixed penalty weights (used instead of neural network predictions)
        # These values are applied uniformly across all time steps
        self.fixed_w_p = 0.1   # Power deviation penalty weight
        self.fixed_w_q = 0.01  # Flow deviation penalty weight
        self.fixed_w_h = 0.05  # Head deviation penalty weight

        # Ablation study with full recursive refinement but fixed weights
        self.max_iterations = 7

    def get_data_file_pattern(self, noise_level=None, random_samples=False):
        """
        Get filename pattern for Ablation training/validation data.

        Args:
            noise_level: Float between 0 and 1 (e.g., 0.1 for 10% noise), or None
            random_samples: Boolean, whether to use random samples dataset

        Returns:
            str: Full path to the ablation data file
        """
        base_dir = self.data_dir
        if random_samples:
            return f"{base_dir}/{self.data_file_prefix}_random_samples.csv"
        elif noise_level is not None:
            noise_pct = int(noise_level * 100)
            return f"{base_dir}/{self.data_file_prefix}_relative_noise_{noise_pct:02d}pct.csv"
        else:
            return f"{base_dir}/{self.data_file_prefix}.csv"
