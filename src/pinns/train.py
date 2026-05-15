"""
Phase 3 — Train the PINN on real spatiotemporal AIS data.

Usage:
    python src/pinns/train.py --location houston --start 2019-01-01 --end 2019-03-31
    python src/pinns/train.py --location houston --start 2019-01-01 --end 2019-06-30 --epochs 3000
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.optim as optim

from src.pinns.lwr_pinn import LWRPINN, total_loss, get_device
from src.pinns.data_prep import get_tensors

log = logging.getLogger(__name__)

MODEL_DIR = Path("outputs/models")
LOG_EVERY = 200
N_COLLOC = 2000


def train(
    location="houston",
    start=date(2019, 1, 1),
    end=date(2019, 3, 31),
    epochs=2000,
    lr=5e-4,
    dx_km=2.0,
    lambda_pde=0.1,
    lambda_kin=0.05,
    constituent_path=None,
    model_name=None,
):
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)

    device = get_device()
    log.info("Device: %s", device)

    log.info("Loading spatiotemporal data: %s %s → %s", location, start, end)
    X, y, meta = get_tensors(
        device,
        start,
        end,
        location,
        dx_km=dx_km,
        constituent_path=constituent_path,
        use_raw_velocity=True,
    )
    log.info("Training points: %d", len(X))

    x_data = X[:, 0:1].detach().requires_grad_(False)
    t_data = X[:, 1:2].detach().requires_grad_(False)
    rho_obs = y[:, 0:1].detach()
    v_obs = y[:, 1:2].detach()

    model = LWRPINN(hidden_layers=4, hidden_size=64).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=200, factor=0.5)

    best_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        opt.zero_grad()

        if epoch % 500 == 1:
            x_col = torch.rand(N_COLLOC, 1, device=device, requires_grad=True)
            t_col = torch.rand(N_COLLOC, 1, device=device, requires_grad=True)
        else:
            x_col = (
                x_col
                if epoch > 1
                else torch.rand(N_COLLOC, 1, device=device, requires_grad=True)
            )
            t_col = (
                t_col
                if epoch > 1
                else torch.rand(N_COLLOC, 1, device=device, requires_grad=True)
            )

        loss, comps = total_loss(
            model,
            x_data,
            t_data,
            rho_obs,
            v_obs,
            x_col,
            t_col,
            lambda_pde=lambda_pde,
            lambda_kin=lambda_kin,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step(loss)

        history.append(comps["total"])
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % LOG_EVERY == 0 or epoch == 1:
            log.info(
                "Epoch %4d/%d  total=%.6f  data=%.6f  pde=%.6f  kin=%.6f",
                epoch,
                epochs,
                comps["total"],
                comps["data"],
                comps["pde"],
                comps["kin"],
            )

    if best_state:
        model.load_state_dict(best_state)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    name = model_name or f"pinn_{location}_{start.isoformat()}_{end.isoformat()}"
    save_path = MODEL_DIR / f"{name}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "location": location,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "best_loss": best_loss,
            "history": history,
            "epochs": epochs,
            "meta": meta,
        },
        save_path,
    )
    log.info("Saved model → %s (best_loss=%.6f)", save_path, best_loss)
    return model, history, meta


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — PINN LWR training")
    parser.add_argument("--location", default="houston")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2019-03-31")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dx-km", type=float, default=2.0)
    parser.add_argument("--lambda-pde", type=float, default=0.1)
    parser.add_argument("--lambda-kin", type=float, default=0.05)
    parser.add_argument("--constituent-path", default=None)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    train(
        location=args.location,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        epochs=args.epochs,
        lr=args.lr,
        dx_km=args.dx_km,
        lambda_pde=args.lambda_pde,
        lambda_kin=args.lambda_kin,
        constituent_path=args.constituent_path,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
