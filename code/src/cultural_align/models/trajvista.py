from __future__ import annotations

import torch
import torch.nn as nn

from cultural_align.data.schema import FEATURE_NAMES
from cultural_align.training.data import NormStats


ACTION_FEATURE_INDICES = (FEATURE_NAMES.index("xAcceleration"), FEATURE_NAMES.index("yAcceleration"))
HIDDEN_DIM = 64
FEATURE_DIM = 64


class SequenceBackbone(nn.Module):
    def __init__(self, input_dim: int = 12):
        super().__init__()
        self.rnn_0 = nn.RNN(input_size=input_dim, hidden_size=32, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size=32, hidden_size=HIDDEN_DIM, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=HIDDEN_DIM, hidden_size=HIDDEN_DIM, batch_first=True)
        self.gru_3 = nn.GRU(input_size=HIDDEN_DIM, hidden_size=HIDDEN_DIM, batch_first=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn_0(state)
        hidden, _ = self.lstm_1(hidden)
        hidden, _ = self.lstm_2(hidden)
        hidden, _ = self.gru_3(hidden)
        return hidden[:, -1]


class HybridPsiCorrectionModel(nn.Module):
    def __init__(
        self,
        stats: NormStats,
        input_dim: int = 12,
        action_dim: int = 2,
        dropout: float = 0.1,
        zero_init_correction: bool = True,
    ):
        super().__init__()
        self.feature_dim = FEATURE_DIM
        self.action_dim = int(action_dim)
        self.backbone = SequenceBackbone(input_dim=input_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        fused = HIDDEN_DIM + 32
        self.phi_head = nn.Sequential(nn.Linear(fused, 128), nn.ReLU(inplace=True), nn.Linear(128, FEATURE_DIM))
        self.psi_head = nn.Sequential(nn.Linear(fused, 128), nn.ReLU(inplace=True), nn.Linear(128, FEATURE_DIM))
        self.film_gamma = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.film_beta = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.ht_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM, action_dim),
        )
        self.psi_correction = nn.Sequential(
            nn.Linear(HIDDEN_DIM + FEATURE_DIM * 3 + FEATURE_DIM, HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN_DIM, action_dim),
        )
        if zero_init_correction:
            nn.init.zeros_(self.psi_correction[-1].weight)
            nn.init.zeros_(self.psi_correction[-1].bias)
        self.w = nn.Parameter(torch.empty(FEATURE_DIM))
        nn.init.normal_(self.w, mean=0.0, std=0.2)
        self.register_buffer("state_mean", torch.tensor(stats.state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.tensor(stats.state_std, dtype=torch.float32))
        self.register_buffer("action_mean", torch.tensor(stats.action_mean, dtype=torch.float32))
        self.register_buffer("action_std", torch.tensor(stats.action_std, dtype=torch.float32))

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.backbone(state)

    def prev_action_norm(self, state: torch.Tensor) -> torch.Tensor:
        prev_norm = state[:, -1, list(ACTION_FEATURE_INDICES)]
        prev_raw = prev_norm * self.state_std[list(ACTION_FEATURE_INDICES)] + self.state_mean[list(ACTION_FEATURE_INDICES)]
        return (prev_raw - self.action_mean) / self.action_std

    def culture_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        gamma = self.film_gamma(self.w).unsqueeze(0)
        beta = self.film_beta(self.w).unsqueeze(0)
        return hidden * (1.0 + gamma) + beta

    def phi_psi_from_hs_action(self, hidden: torch.Tensor, action_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_hidden = self.action_encoder(action_norm)
        fused = torch.cat([hidden, action_hidden], dim=-1)
        return self.phi_head(fused), self.psi_head(fused)

    def phi_psi_q_from_hs_action(self, hidden: torch.Tensor, action_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi, psi = self.phi_psi_from_hs_action(hidden, action_norm)
        q = (psi * self.w.unsqueeze(0)).sum(dim=-1)
        return phi, psi, q

    def forward_sa(self, state: torch.Tensor, action_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.phi_psi_q_from_hs_action(self.encode_state(state), action_norm)

    def q_on_action_set_from_hs(self, hidden: torch.Tensor, actions_norm: torch.Tensor) -> torch.Tensor:
        batch, num_actions, _ = actions_norm.shape
        hidden_rep = hidden.unsqueeze(1).expand(batch, num_actions, hidden.shape[-1]).reshape(batch * num_actions, -1)
        action_hidden = self.action_encoder(actions_norm.reshape(batch * num_actions, self.action_dim))
        psi = self.psi_head(torch.cat([hidden_rep, action_hidden], dim=-1))
        q = (psi * self.w.unsqueeze(0)).sum(dim=-1)
        return q.view(batch, num_actions)

    def q_on_action_set(self, state: torch.Tensor, actions_norm: torch.Tensor) -> torch.Tensor:
        return self.q_on_action_set_from_hs(self.encode_state(state), actions_norm)

    def forward_from_hs(self, hidden: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        prev_norm = self.prev_action_norm(state)
        phi, psi = self.phi_psi_from_hs_action(hidden, prev_norm)
        w = self.w.unsqueeze(0).expand(hidden.shape[0], -1)
        ht_residual = self.ht_head(self.culture_hidden(hidden))
        psi_residual = self.psi_correction(torch.cat([hidden, phi, psi, psi * w, w], dim=-1))
        return prev_norm + ht_residual + psi_residual

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward_from_hs(self.encode_state(state), state)


def make_model(
    stats: NormStats,
    input_dim: int = 12,
    dropout: float = 0.1,
) -> HybridPsiCorrectionModel:
    return HybridPsiCorrectionModel(
        stats,
        input_dim=input_dim,
        dropout=dropout,
    )
