"""Approximate unlearning methods for the era pilot.

Each takes a trained model (mutated in place), returns a result dict with
wall-clock seconds and a per-epoch trajectory so over/under-forgetting is
visible in the metrics JSON. Model selection is deliberately naive (final
epoch) for the ascent methods: at deployment there is no oracle to select
against, and best-val selection would simply pick the least-unlearned state.
"""
from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from fin_ts_ptq.training.loop import TrainConfig, evaluate_loss, train_model


def finetune_retain(
    model: nn.Module,
    retain_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    patience: int = 5,
) -> dict[str, Any]:
    """Catastrophic-forgetting baseline: keep training on the retain set only."""
    t0 = time.perf_counter()
    res = train_model(
        model,
        retain_loader,
        val_loader,
        TrainConfig(epochs=epochs, lr=lr, weight_decay=weight_decay, patience=patience, log_every=1),
        device,
    )
    return {
        "seconds": time.perf_counter() - t0,
        "epochs_ran": res.epochs_ran,
        "best_val_loss": res.best_val_loss,
    }


def neggrad_plus(
    model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    alpha: float = 0.5,
    hinge_margin: float | None = None,
) -> dict[str, Any]:
    """NegGrad+: descend on retain, ascend on forget.

    hinge_margin=None -> plain `L_retain - alpha * L_forget` (unbounded
    ascent; expect divergence, that is part of the measurement).
    hinge_margin=m -> `L_retain + alpha * relu(m - L_forget)`: push the
    forget loss up to m (the uninformative level; targets are
    cross-sectionally z-scored so an ignorant predictor scores MSE ~= 1)
    and no further.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    t0 = time.perf_counter()
    trajectory: list[dict[str, float]] = []
    forget_iter = iter(forget_loader)

    for epoch in range(epochs):
        model.train()
        retain_sum = retain_count = 0.0
        for xr, yr in retain_loader:
            try:
                xf, yf = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                xf, yf = next(forget_iter)
            xr, yr = xr.to(device), yr.to(device)
            xf, yf = xf.to(device), yf.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_r = mse(model(xr), yr)
            loss_f = mse(model(xf), yf)
            if hinge_margin is None:
                loss = loss_r - alpha * loss_f
            else:
                loss = loss_r + alpha * torch.relu(hinge_margin - loss_f)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            retain_sum += float(loss_r.item()) * xr.shape[0]
            retain_count += xr.shape[0]
        epoch_stats = {
            "epoch": epoch + 1,
            "retain_loss": retain_sum / max(1.0, retain_count),
            "forget_loss": evaluate_loss(model, forget_loader, mse, device),
            "val_loss": evaluate_loss(model, val_loader, mse, device),
        }
        trajectory.append(epoch_stats)
        print(f"[NEGGRAD] {epoch_stats}", flush=True)

    return {
        "seconds": time.perf_counter() - t0,
        "epochs_ran": epochs,
        "alpha": alpha,
        "hinge_margin": hinge_margin,
        "trajectory": trajectory,
    }


def scrub(
    model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    msteps: int = 3,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    gamma: float = 1.0,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """SCRUB (Kurmanji et al., arXiv:2302.09880) adapted to regression.

    Teacher = frozen copy of the input model. The classification KL terms
    become squared distances between student and teacher outputs: for the
    first `msteps` epochs, a max-pass pushes the student AWAY from the
    teacher on the forget set, then every epoch a min-pass pulls it toward
    the teacher on retain (weight alpha) plus the task loss (weight gamma).
    """
    import copy as _copy

    teacher = _copy.deepcopy(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    t0 = time.perf_counter()
    trajectory: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        if epoch < msteps:
            for xf, _ in forget_loader:
                xf = xf.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = -mse(model(xf), teacher(xf))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        for xr, yr in retain_loader:
            xr, yr = xr.to(device), yr.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xr)
            loss = alpha * mse(pred, teacher(xr)) + gamma * mse(pred, yr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        epoch_stats = {
            "epoch": epoch + 1,
            "max_pass": epoch < msteps,
            "forget_loss": evaluate_loss(model, forget_loader, mse, device),
            "val_loss": evaluate_loss(model, val_loader, mse, device),
        }
        trajectory.append(epoch_stats)
        print(f"[SCRUB] {epoch_stats}", flush=True)

    return {
        "seconds": time.perf_counter() - t0,
        "epochs_ran": epochs,
        "msteps": msteps,
        "gamma": gamma,
        "alpha": alpha,
        "trajectory": trajectory,
    }


def ts_unlearn(
    model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
) -> dict[str, Any]:
    """Faithful TS-Unlearn (Wang et al., PAKDD 2026, DOI 10.1007/978-981-92-1462-4_27).

    Teacher = frozen copy of the input model. Retain loss distills to the
    teacher (eq. 8, NOT ground-truth labels); forget loss pulls student
    outputs toward the teacher's response to Gaussian-noise INPUTS (eq. 9,
    n ~ N(0, I) — inputs are z-scored so identity covariance matches their
    normalized-input assumption); per-step alpha from the analytic MGDA
    min-norm solution (eq. 12). Two backward passes per step.
    """
    import copy as _copy

    teacher = _copy.deepcopy(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    t0 = time.perf_counter()
    trajectory: list[dict[str, float]] = []
    forget_iter = iter(forget_loader)

    for epoch in range(epochs):
        model.train()
        alpha_sum = alpha_n = 0.0
        for xr, _ in retain_loader:
            try:
                xf, _ = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                xf, _ = next(forget_iter)
            xr, xf = xr.to(device), xf.to(device)
            with torch.no_grad():
                target_r = teacher(xr)
                target_f = teacher(torch.randn_like(xf))
            loss_r = mse(model(xr), target_r)
            loss_f = mse(model(xf), target_f)
            # allow_unused: some archs (TimeMixer) have parameters outside
            # the forward graph; treat their gradient as zero
            def flat_grad(loss):
                gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                return torch.cat([
                    (g if g is not None else torch.zeros_like(p)).reshape(-1)
                    for g, p in zip(gs, params)
                ])
            g_f = flat_grad(loss_f)
            g_r = flat_grad(loss_r)
            diff = g_r - g_f
            alpha = float(torch.clamp(torch.dot(diff, g_r) / (diff.dot(diff) + 1e-12), 0.0, 1.0))
            optimizer.zero_grad(set_to_none=True)
            (alpha * loss_f + (1.0 - alpha) * loss_r).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            alpha_sum += alpha
            alpha_n += 1
        epoch_stats = {
            "epoch": epoch + 1,
            "mean_alpha": alpha_sum / max(1.0, alpha_n),
            "forget_loss": evaluate_loss(model, forget_loader, mse, device),
            "val_loss": evaluate_loss(model, val_loader, mse, device),
        }
        trajectory.append(epoch_stats)
        print(f"[TSUNLEARN] {epoch_stats}", flush=True)

    return {
        "seconds": time.perf_counter() - t0,
        "epochs_ran": epochs,
        "trajectory": trajectory,
    }


def relabel_noise(
    model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    alpha: float = 0.5,
    noise_scale: float = 1.0,
) -> dict[str, Any]:
    """Noise-relabel: fine-tune with forget targets replaced by Gaussian draws.

    Stand-in for TS-Unlearn's objective (PAKDD'26; abstract-only until we
    obtain the full text): forgotten predictions should converge to a noise
    distribution. Targets here are cross-sectionally z-scored, so N(0,
    noise_scale^2) is the marginal-matched ignorant target.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    t0 = time.perf_counter()
    trajectory: list[dict[str, float]] = []
    forget_iter = iter(forget_loader)

    for epoch in range(epochs):
        model.train()
        for xr, yr in retain_loader:
            try:
                xf, yf = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                xf, yf = next(forget_iter)
            xr, yr = xr.to(device), yr.to(device)
            xf = xf.to(device)
            noise_targets = torch.randn_like(yf).to(device) * noise_scale
            optimizer.zero_grad(set_to_none=True)
            loss = mse(model(xr), yr) + alpha * mse(model(xf), noise_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        epoch_stats = {
            "epoch": epoch + 1,
            "forget_loss": evaluate_loss(model, forget_loader, mse, device),
            "val_loss": evaluate_loss(model, val_loader, mse, device),
        }
        trajectory.append(epoch_stats)
        print(f"[RELABEL] {epoch_stats}", flush=True)

    return {
        "seconds": time.perf_counter() - t0,
        "epochs_ran": epochs,
        "alpha": alpha,
        "noise_scale": noise_scale,
        "trajectory": trajectory,
    }
