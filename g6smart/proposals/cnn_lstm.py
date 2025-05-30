import torch
import torch.nn as nn


class CLAllocator(nn.Module):
    def __init__(
        self, n_subnetworks: int, n_subbands: int,
        feature_dim: int = 128, lstm_dim: int = 256, lstm_layers: int = 3
    ) -> None:
        super().__init__()

        assert n_subbands >= 4, "the model config only works with at least 4 bands"
        assert n_subnetworks >= 4, "the model config only works with at least 4 bands"

        self.N = n_subnetworks
        self.K = n_subbands

        self.feature_dim = feature_dim
        self.hidden_dim  = lstm_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(), # GELU generally performs well, keep unless direct performance hit
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(16, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, self.feature_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.lstm = nn.LSTM(
            input_size    = self.feature_dim,
            hidden_size   = self.hidden_dim,
            num_layers    = lstm_layers,
            batch_first   = True,
            bidirectional = True # Keep bidirectional for potential performance unless proven slower
        )

        lstm_output_dim = self.hidden_dim * 2 if self.lstm.bidirectional else self.hidden_dim

        self.allocator = nn.Sequential(
            nn.Linear(lstm_output_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.Dropout(p = 0.2),

            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(p = 0.2),

            nn.Linear(self.hidden_dim, self.K),
            nn.Softmax(dim = 1) # Keep Softmax only if loss_pure_rate expects probabilities, not logits
        )

    def preprocess(self, H: torch.Tensor) -> torch.Tensor:
        Hdb = 10 * torch.log10(H + 1e-12)
        mean, std = -70, -10 # empirical metrics
        return (Hdb - mean) / std

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        batch_size = H.shape[0]
        H = self.preprocess(H)
        H_reshaped_for_encoder = H.permute(0, 2, 1, 3).reshape(batch_size * self.N, self.K, self.N).unsqueeze(1)

        all_encodings = self.encoder(H_reshaped_for_encoder)
        encodings = all_encodings.reshape(batch_size, self.N, self.feature_dim)

        lstm_output, _ = self.lstm(encodings)

        lstm_output_reshaped = lstm_output.reshape(-1, lstm_output.shape[-1])
        all_probs = self.allocator(lstm_output_reshaped)

        probs = all_probs.reshape(batch_size, self.N, self.K)
        probs = probs.permute(0, 2, 1)
        return probs
