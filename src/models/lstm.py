"""LSTM sequence model that forecasts the next `horizon` OHLC candles.

Targets are fractional offsets from the anchor close (see
`features.to_relative_targets`), so the model learns shape rather than price
level. Uncertainty comes from Monte Carlo dropout: dropout stays active at
inference and repeated passes give a predictive distribution per candle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CandleLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = self.dropout(out[:, -1, :])
        return self.head(last).view(-1, self.horizon, 4)


def train(
    model: CandleLSTM,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 6,
    device: str | None = None,
) -> dict:
    """Train with early stopping on validation loss. Returns the run history."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Huber is less swayed by the fat tails that price data always has.
    criterion = nn.HuberLoss(delta=0.01)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, stale = float("inf"), None, 0

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(xv), yv).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    return history


@torch.no_grad()
def predict(
    model: CandleLSTM,
    x: np.ndarray,
    mc_samples: int = 50,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo dropout prediction.

    Returns (mean, std), each shaped (n_samples, horizon, 4), in relative-target
    space. The std is the model's epistemic spread — it is NOT a calibrated
    confidence interval and should be presented as a spread, not a guarantee.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # Re-enable dropout only; batchnorm/LSTM stay in eval mode.
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    xt = torch.from_numpy(x).to(device)
    draws = torch.stack([model(xt) for _ in range(mc_samples)])
    return draws.mean(0).cpu().numpy(), draws.std(0).cpu().numpy()


def save(model: CandleLSTM, path: str | Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load(path: str | Path) -> tuple[CandleLSTM, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob["meta"]
    model = CandleLSTM(
        n_features=meta["n_features"],
        horizon=meta["horizon"],
        hidden_size=meta.get("hidden_size", 64),
        num_layers=meta.get("num_layers", 2),
        dropout=meta.get("dropout", 0.2),
    )
    model.load_state_dict(blob["state_dict"])
    return model, meta
