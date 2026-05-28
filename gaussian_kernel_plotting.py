# Experimentation: Plotting MC vs. learnt Pt error distributions.
# Feb-2026

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gaussian_kernel_single import make_kernel
from gaussian_kernel_variable_alpha import make_mixture_kernel


INPUT_DIR = Path("PtErrSC")
CKPT_SC = "kernel_SC/gaussian_mixture/version_5/checkpoints/epoch=70-step=1349.ckpt"

OUTPUT_FILE = Path("smearing_plots/pt_err_real_vs_kernel_sc.png")
COLUMN = "pt_err"
BINS = 5000
BATCH_SIZE = 200_000
USE_INVERSE = False
XRANGE = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_folder(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {folder}")
    return pd.concat((pd.read_pickle(f) for f in files), ignore_index=True)


def load_kernel(ckpt_path: str):
    kernel = make_mixture_kernel(4000, 200 ,8000, 0, 0.5,True)
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    kernel.load_state_dict(ckpt_dict["state_dict"])
    kernel.to(device).eval()
    return kernel


def sample_kernel_pt_err(df: pd.DataFrame, kernel, batch_size: int = BATCH_SIZE) -> np.ndarray:
    pt_np = np.array(df["pt"].to_numpy(np.float32), copy=True)
    eta_np = np.array(df["eta"].to_numpy(np.float32), copy=True)
    out = np.empty(len(df), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            pt = torch.from_numpy(pt_np[start:end]).to(device)
            eta = torch.from_numpy(eta_np[start:end]).to(device)
            x = torch.stack([pt, eta], dim=1)

            err = kernel.draw(x)
            if err.ndim == 2:
                err = err[:, 0]

            out[start:end] = err.detach().cpu().numpy()

            if device.type == "cuda":
                torch.cuda.empty_cache()

    return out


def transform_values(values: np.ndarray, use_inverse: bool) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(vals)
    if use_inverse:
        mask &= vals != 0.0
    vals = vals[mask]
    if use_inverse:
        vals = 1.0 / vals
        vals = vals[np.isfinite(vals)]
    return vals



def main() -> None:
    df = load_folder(INPUT_DIR)
    needed = {"pt", "eta", COLUMN}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    real_pt_err = pd.to_numeric(df[COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    kernel_sc = load_kernel(CKPT_SC)
    kernel_pt_err = sample_kernel_pt_err(df, kernel_sc)

    real_values = transform_values(real_pt_err, USE_INVERSE)
    kernel_values = transform_values(kernel_pt_err, USE_INVERSE)
    if real_values.size == 0:
        raise ValueError("No finite real values left after transforms.")
    if kernel_values.size == 0:
        raise ValueError("No finite kernel values left after transforms.")

    print("files:", len(list(INPUT_DIR.glob("*.pkl"))))

    combined = np.concatenate([real_values, kernel_values])
    vmin = float(np.min(combined))
    vmax = float(np.max(combined))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Histogram range is not finite.")
    if vmin == vmax:
        eps = 1e-6 if vmin == 0.0 else abs(vmin) * 1e-6
        vmin -= eps
        vmax += eps
    bin_edges = np.linspace(vmin, vmax, BINS + 1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.hist(real_values, bins=bin_edges, histtype="step", linewidth=1.6, density=True, label="MC")
    plt.hist(kernel_values, bins=bin_edges, histtype="step", linewidth=1.6, density=True, label="Kernel draw")
    plt.xlabel(f"1/{COLUMN}" if USE_INVERSE else COLUMN)
    if XRANGE is not None:
        plt.xlim(*XRANGE)
    plt.ylabel("density")
    plt.legend()
    plt.title(f"MC vs. Kernel Pt Error distribution for Flipped Electrons")
    plt.tight_layout()
    plt.xlim(-20000,20000)
    plt.savefig(OUTPUT_FILE, dpi=160)
    plt.close()
    print(f"Saved histogram to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
