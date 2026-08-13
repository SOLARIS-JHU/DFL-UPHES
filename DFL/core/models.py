"""
Neural network models for weight prediction.

This module contains models that predict penalty weights for the optimization:
- BoundedLogWeightPredictor: LSTM/RNN/FC/BILSTM/CNN/TRANSFORMER network predicting weights in log-domain
- FixedWeightConfig: Fixed weights for ablation studies (no neural network)
"""

import torch
import torch.nn as nn


class BoundedLogWeightPredictor(nn.Module):
    """
    Modified weight predictor with bounded log-domain weights.

    Predicts penalty weights (w_p, w_q, w_h) for the optimization problem
    using an LSTM, RNN, or FC architecture. Works in log-domain to ensure
    positive weights and better numerical stability.
    """

    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.2,
                 time_horizon=24, archetype='LSTM',
                 init_w_p=0.05, init_w_q=0.05, init_w_h=0.05,
                 w_p_min=0.01, w_p_max=10.0,
                 w_q_min=0.01, w_q_max=5.0,
                 w_h_min=0.01, w_h_max=5.0):
        """
        Initialize BoundedLogWeightPredictor.

        Args:
            input_size: Number of input features (default 4: price, power, flow, head)
            hidden_size: Hidden layer size
            num_layers: Number of recurrent layers
            dropout: Dropout rate (only applied if num_layers > 1)
            time_horizon: Number of time periods (default 24 hours)
            archetype: Network architecture ('LSTM', 'RNN', 'FC', 'BILSTM', 'CNN', or 'TRANSFORMER')
            init_w_p, init_w_q, init_w_h: Initial weight values
            w_p_min, w_p_max: Bounds for power deviation penalty weight
            w_q_min, w_q_max: Bounds for flow deviation penalty weight
            w_h_min, w_h_max: Bounds for head deviation penalty weight
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_horizon = time_horizon
        self.archetype = archetype.upper()

        # Process initial weights
        self.init_w_p = init_w_p
        self.init_w_q = init_w_q
        self.init_w_h = init_w_h

        # Store bounds for each weight type
        self.w_p_min = w_p_min
        self.w_p_max = w_p_max
        self.w_q_min = w_q_min
        self.w_q_max = w_q_max
        self.w_h_min = w_h_min
        self.w_h_max = w_h_max

        # Compute log-domain bounds
        self.log_w_p_min = torch.log(torch.tensor(w_p_min))
        self.log_w_p_max = torch.log(torch.tensor(w_p_max))
        self.log_w_q_min = torch.log(torch.tensor(w_q_min))
        self.log_w_q_max = torch.log(torch.tensor(w_q_max))
        self.log_w_h_min = torch.log(torch.tensor(w_h_min))
        self.log_w_h_max = torch.log(torch.tensor(w_h_max))

        # Same architecture as before
        if self.archetype == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.archetype == 'RNN':
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.archetype == 'BILSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
            self.bidirectional_proj = nn.Linear(hidden_size * 2, hidden_size)
        elif self.archetype == 'CNN':
            channels = [32, 64, hidden_size]
            cnn_layers = []
            in_ch = input_size
            for i, out_ch in enumerate(channels):
                cnn_layers += [
                    nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                ]
                if i < len(channels) - 1:
                    cnn_layers.append(nn.Dropout(dropout))
                in_ch = out_ch
            self.cnn = nn.Sequential(*cnn_layers)
            self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        elif self.archetype == 'TRANSFORMER':
            d_model = hidden_size
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, time_horizon, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, activation='gelu'
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.transformer_norm = nn.LayerNorm(d_model)
        elif self.archetype == 'FC':
            self.fc_layers = nn.Sequential(
                nn.Linear(input_size * time_horizon, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unsupported archetype: {archetype}. "
                             f"Choose from 'LSTM', 'RNN', 'FC', 'BILSTM', 'CNN', or 'TRANSFORMER'.")

        # Output layer for log-weights
        self.output = nn.Linear(hidden_size, 3 * time_horizon)

        # Initialize remaining weights
        self._init_weights()

        # Set initial biases using the provided weight values
        self._set_initial_weights()

    def _init_weights(self):
        """Initialize weights based on the selected architecture"""
        for name, param in self.named_parameters():
            if 'pos_encoding' in name or 'norm' in name or 'bn' in name:
                continue
            if 'weight' in name and 'output' not in name:
                if param.dim() >= 2:
                    nn.init.xavier_normal_(param, gain=1.5)
            elif 'bias' in name and 'output' not in name:
                nn.init.constant_(param, 0.1)

    def _set_initial_weights(self):
        """Set the output bias to initialize log-weights to desired values"""
        # Convert weight values to log domain
        log_w_p = torch.log(torch.tensor(self.init_w_p))
        log_w_q = torch.log(torch.tensor(self.init_w_q))
        log_w_h = torch.log(torch.tensor(self.init_w_h))

        # The bias has shape [3 * time_horizon]
        # We need to set segments of it for each weight type
        bias = self.output.bias.data

        # Set the first third to log_w_p
        bias[0:self.time_horizon] = log_w_p

        # Set the middle third to log_w_q
        bias[self.time_horizon:2*self.time_horizon] = log_w_q

        # Set the last third to log_w_h
        bias[2*self.time_horizon:3*self.time_horizon] = log_w_h

    def _clamp_log_weights(self, log_w_p, log_w_q, log_w_h):
        """Clamp log weights to ensure they stay within bounds"""
        # Move bounds to the appropriate device
        device = log_w_p.device
        log_w_p_min = self.log_w_p_min.to(device)
        log_w_p_max = self.log_w_p_max.to(device)
        log_w_q_min = self.log_w_q_min.to(device)
        log_w_q_max = self.log_w_q_max.to(device)
        log_w_h_min = self.log_w_h_min.to(device)
        log_w_h_max = self.log_w_h_max.to(device)

        # Apply clamping
        log_w_p = torch.clamp(log_w_p, min=log_w_p_min, max=log_w_p_max)
        log_w_q = torch.clamp(log_w_q, min=log_w_q_min, max=log_w_q_max)
        log_w_h = torch.clamp(log_w_h, min=log_w_h_min, max=log_w_h_max)

        return log_w_p, log_w_q, log_w_h

    def forward(self, x):
        # Same forward logic as before
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if self.archetype in ['LSTM', 'RNN']:
            if self.archetype == 'LSTM':
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
                output, _ = self.rnn(x, (h0, c0))
            else:  # RNN
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
                output, _ = self.rnn(x, h0)

            last_output = output[:, -1, :]
        elif self.archetype == 'BILSTM':
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
            output, _ = self.rnn(x, (h0, c0))
            last_output = self.bidirectional_proj(output[:, -1, :])
        elif self.archetype == 'CNN':
            # x: [batch, seq_len, features] -> [batch, features, seq_len]
            x_cnn = x.transpose(1, 2)
            x_cnn = self.cnn(x_cnn)
            last_output = self.cnn_pool(x_cnn).squeeze(-1)
        elif self.archetype == 'TRANSFORMER':
            x_proj = self.input_proj(x) + self.pos_encoding
            x_enc = self.transformer_encoder(x_proj)
            x_enc = self.transformer_norm(x_enc)
            last_output = x_enc.mean(dim=1)
        else:  # FC architecture
            batch_size = x.size(0)
            x_flat = x.reshape(batch_size, -1)
            last_output = self.fc_layers(x_flat)

        # Get log-weights through output layer (no activation)
        log_weights = self.output(last_output)

        # Reshape log-weights
        log_weights = log_weights.view(-1, 3, self.time_horizon)
        log_w_p, log_w_q, log_w_h = log_weights[:, 0, :], log_weights[:, 1, :], log_weights[:, 2, :]

        # Apply bounds to log weights
        log_w_p, log_w_q, log_w_h = self._clamp_log_weights(log_w_p, log_w_q, log_w_h)

        # Remove batch dimension if it was added
        if x.size(0) == 1:
            log_w_p, log_w_q, log_w_h = log_w_p.squeeze(0), log_w_q.squeeze(0), log_w_h.squeeze(0)

        return log_w_p, log_w_q, log_w_h

    def predict_weights(self, DA_prices, power, flow, head):
        """
        Predict weights for a sequence of inputs.
        Returns both log_weights and exponentiated weights.
        """
        # Stack features into sequence
        x = torch.stack([DA_prices, power, flow, head], dim=1)  # [time_horizon, 4]

        with torch.no_grad():
            log_w_p, log_w_q, log_w_h = self.forward(x)

            # Exponentiate to get actual weights
            w_p = torch.exp(log_w_p)
            w_q = torch.exp(log_w_q)
            w_h = torch.exp(log_w_h)

        return (log_w_p, log_w_q, log_w_h), (w_p, w_q, w_h)


class FixedWeightConfig:
    """
    Configuration for fixed penalty weights (no neural network).

    Used in ablation studies to test the impact of removing the
    neural network component while keeping the recursive linearization.
    """

    def __init__(self, w_p=0.6, w_q=0.02, w_h=0.1, time_horizon=24, device=None):
        """
        Initialize FixedWeightConfig.

        Args:
            w_p: Fixed power deviation penalty weight
            w_q: Fixed flow deviation penalty weight
            w_h: Fixed head deviation penalty weight
            time_horizon: Number of time periods
            device: PyTorch device (cpu or cuda)
        """
        self.w_p_base = w_p
        self.w_q_base = w_q
        self.w_h_base = w_h
        self.time_horizon = time_horizon
        self.device = device if device is not None else torch.device("cpu")

    def get_weights(self):
        """Return fixed weights as tensors"""
        w_p = torch.full((self.time_horizon,), self.w_p_base, dtype=torch.float32, device=self.device)
        w_q = torch.full((self.time_horizon,), self.w_q_base, dtype=torch.float32, device=self.device)
        w_h = torch.full((self.time_horizon,), self.w_h_base, dtype=torch.float32, device=self.device)
        return w_p, w_q, w_h
