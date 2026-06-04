from __future__ import annotations

import torch


def bc_q_loss(model, state: torch.Tensor, action: torch.Tensor, k_neg: int, neg_std: float) -> torch.Tensor:
    if k_neg <= 0:
        return torch.zeros((), dtype=action.dtype, device=action.device)
    negs = action.unsqueeze(1) + torch.randn(action.shape[0], k_neg, action.shape[1], device=action.device, dtype=action.dtype) * neg_std
    actions = torch.cat([action.unsqueeze(1), negs], dim=1)
    q = model.q_on_action_set(state, actions)
    logp = q - torch.logsumexp(q, dim=1, keepdim=True)
    return -logp[:, 0].mean()


def bc_q_loss_from_hs(model, hs: torch.Tensor, action: torch.Tensor, k_neg: int, neg_std: float) -> torch.Tensor:
    if k_neg <= 0:
        return torch.zeros((), dtype=action.dtype, device=action.device)
    negs = action.unsqueeze(1) + torch.randn(action.shape[0], k_neg, action.shape[1], device=action.device, dtype=action.dtype) * neg_std
    actions = torch.cat([action.unsqueeze(1), negs], dim=1)
    q = model.q_on_action_set_from_hs(hs, actions)
    logp = q - torch.logsumexp(q, dim=1, keepdim=True)
    return -logp[:, 0].mean()


def itd_loss(model, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, next_action: torch.Tensor, gamma: float) -> torch.Tensor:
    phi, psi, _q = model.forward_sa(state, action)
    with torch.no_grad():
        psi_next = model.forward_sa(next_state, next_action)[1]
    return (psi - (phi + gamma * psi_next)).pow(2).mean()


def itd_loss_from_hs(model, hs: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, next_action: torch.Tensor, gamma: float) -> torch.Tensor:
    phi, psi, _q = model.phi_psi_q_from_hs_action(hs, action)
    with torch.no_grad():
        next_hs = model.encode_state(next_state)
        psi_next = model.phi_psi_q_from_hs_action(next_hs, next_action)[1]
    return (psi - (phi + gamma * psi_next)).pow(2).mean()

