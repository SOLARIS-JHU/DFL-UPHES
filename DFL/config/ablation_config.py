"""
Ablation variant configuration (No Neural Network).

This configuration tests the recursive linearization approach without
the neural network component, using fixed penalty weights instead.
"""

from .pw_config import PWConfig


class AblationConfig(PWConfig):
    """
    Configuration for Ablation variant (No Neural Network).

    Inherits from PWConfig since it uses the same data source (piecewise MIQP),
    but disables the neural network and uses fixed weights instead.
    """

    def __init__(self):
        """Initialize Ablation configuration."""
        super().__init__()
        self.variant_name = "Ablation"

        # Key difference: disable neural network
        self.use_neural_network = False

        # Use fixed weights for all penalties
        # These values are used consistently across all time steps
        self.fixed_w_p = 0.6   # Power deviation penalty weight
        self.fixed_w_q = 0.02  # Flow deviation penalty weight
        self.fixed_w_h = 0.1   # Head deviation penalty weight
