"""
Base configuration class for DFL variants.

This module defines the base configuration class that all variant-specific
configurations inherit from.
"""

import os


class DFLConfig:
    """
    Base configuration for DFL (Deep Federated Learning) variants.

    Defines common parameters and methods that all variants (GL, PW, Ablation) share.
    Subclasses should override methods to provide variant-specific behavior.
    """

    def __init__(self):
        """Initialize base configuration with default values."""
        # Variant identifier
        self.variant_name = "base"

        # Base paths (relative to repository root)
        self.miqp_base_path = "./MIQP"
        self.library_path = "./Library"
        self.preprocess_file = "./preprocess.pkl"

        # Data file prefixes and patterns (to be set by subclasses)
        self.data_file_prefix = None
        self.data_file_pattern = None  # Pattern with {noise} placeholder
        self.random_samples_file = None  # File path for random samples data

        # Neural network settings
        self.use_neural_network = True
        self.architecture = 'LSTM'  # Options: 'LSTM', 'RNN', 'FC'
        self.num_layers = 3
        self.hidden_size = 128
        self.dropout = 0.2

        # Weight initialization and bounds for neural network training
        self.init_w_p = 0.6      # Initial power deviation penalty weight
        self.init_w_q = 0.02     # Initial flow deviation penalty weight
        self.init_w_h = 0.1      # Initial head deviation penalty weight

        # Bounds for learned penalty weights
        self.w_p_min = 0.1
        self.w_p_max = 3.0
        self.w_q_min = 0.001
        self.w_q_max = 0.2
        self.w_h_min = 0.01
        self.w_h_max = 5.0

        # Fixed weight settings for baseline (non-neural network) mode
        self.fixed_w_p = 0.6
        self.fixed_w_q = 0.02
        self.fixed_w_h = 0.1

        # Training settings
        self.max_iterations = 7           # Recursive linearization iterations
        self.penalty_growth_rate = 1.5    # Growth factor for penalty weights per iteration
        self.learning_rate = 0.001        # Adam optimizer learning rate
        self.num_epochs = 500             # Maximum training epochs
        self.patience = 20                # Early stopping patience

        # Optimization settings
        self.time_horizon = 24  # Hours
        self.sampling_rate = 50

        # System parameters
        self.δ_p = 0.5
        self.δ_h = 1.0
        self.δ_q = 0.5
        self.operational_cost = 0.4

        # Output directories (centralized in DFL/outputs/)
        self.outputs_root = "./DFL/outputs"            # Root for all outputs
        self.data_dir = "./DFL/outputs/noisy_data"     # Directory for generated noisy data
        self.output_base_dir = "./DFL/outputs/trained_models"  # Directory for trained models
        self.results_base_dir = "./DFL/outputs/validation_results"  # Directory for validation results

    def get_miqp_file_path(self):
        """
        Get path to original MIQP results file.

        Returns:
            str: Full path to MIQP results CSV file

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_miqp_file_path()")

    def get_data_file_pattern(self, noise_level=None, random_samples=False):
        """
        Get filename pattern for training/validation data.

        Args:
            noise_level: Float between 0 and 1 (e.g., 0.1 for 10% noise), or None
            random_samples: Boolean, whether to use random samples dataset

        Returns:
            str: Filename for the data file

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_data_file_pattern()")

    def get_output_dir(self, source_name, config_name):
        """
        Get output directory for trained models.

        Args:
            source_name: Data source name (e.g., "MIQP_linear_results_random_samples")
            config_name: Configuration name (e.g., "LSTM_3layer_10iter")

        Returns:
            str: Path to output directory
        """
        return os.path.join(self.output_base_dir, source_name, config_name)

    def get_results_file(self, noise_level=None, random_samples=False):
        """
        Get results CSV file path for generated noisy data.

        Args:
            noise_level: Float between 0 and 1, or None
            random_samples: Boolean, whether to use random samples dataset

        Returns:
            str: Path to results CSV file in data_dir
        """
        full_path = self.get_data_file_pattern(noise_level, random_samples)
        # Extract just the filename from the full path (in case it includes directory prefixes)
        filename = os.path.basename(full_path)
        return os.path.join(self.data_dir, filename)

    def get_model_config_name(self):
        """
        Get model configuration name for directory naming.

        Returns:
            str: Configuration name (e.g., "LSTM_3layer_3iter")
        """
        if self.use_neural_network:
            return f"{self.architecture}_{self.num_layers}layer_{self.max_iterations}iter"
        else:
            return f"NoNN_{self.max_iterations}iter"

    def __repr__(self):
        """String representation of the configuration."""
        return (f"DFLConfig(variant={self.variant_name}, "
                f"architecture={self.architecture if self.use_neural_network else 'NoNN'}, "
                f"layers={self.num_layers}, iterations={self.max_iterations})")
