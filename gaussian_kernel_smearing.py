from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from gaussian_kernel_single import make_kernel
from gaussian_kernel_variable_alpha import make_mixture_kernel

# plot kernel smearing sampled from truth + reco. 

device = torch.device( "cpu")

CKPT_OC = "kernel_OC/gaussian/version_66/checkpoints/epoch=8-step=13734.ckpt"
CKPT_SC = "kernel_SC/gaussian_mixture/version_5/checkpoints/epoch=70-step=1349.ckpt"

def load_folder(folder: str, fallback_folders: tuple[str, ...] = ()) -> tuple[pd.DataFrame, str]:
    checked: list[str] = []
    candidates = (folder, *fallback_folders)

    for idx, candidate in enumerate(candidates):
        folder_path = Path(candidate)
        if not folder_path.exists():
            checked.append(f"{candidate} (missing)")
            continue

        files = sorted(folder_path.glob("*.pkl"))
        if not files:
            checked.append(f"{candidate} (no .pkl files)")
            continue

        if idx > 0:
            print(f"Using fallback dataset folder: {candidate}")

        dfs = [pd.read_pickle(f) for f in files]
        return pd.concat(dfs, ignore_index=True), candidate

    details = ", ".join(checked)
    raise FileNotFoundError(f"No readable dataset folder found. Checked: {details}")


def validate_required_columns(df: pd.DataFrame, name: str, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")

def load_kernel(ckpt_path: str):
    k = make_kernel(4000, 200)
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    k.load_state_dict(ckpt_dict["state_dict"])
    k.to(device).eval()
    return k

def load_kernel_mixture(ckpt_path: str):
    k = make_mixture_kernel(4000, 200, 8000, 0, 0.5, True)
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    k.load_state_dict(ckpt_dict["state_dict"])
    k.to(device).eval()
    return k

def kernel_error_mean_std(kernel, x: torch.Tensor):
    if hasattr(kernel, "sub_kernels") and hasattr(kernel, "log_probability_model"):
        log_probs = kernel.log_probability_model(x)
        probs = torch.exp(log_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        alpha = probs[..., 0].clamp(0.0, 1.0)
        choose_first = torch.rand_like(alpha) < alpha

        mean_1 = kernel.sub_kernels[0].loc_model(x).squeeze(-1)
        std_1 = kernel.sub_kernels[0].scale_model(x).squeeze(-1).clamp_min(1e-8)
        mean_2 = kernel.sub_kernels[1].loc_model(x).squeeze(-1)
        std_2 = kernel.sub_kernels[1].scale_model(x).squeeze(-1).clamp_min(1e-8)

        mean = torch.where(choose_first, mean_1, mean_2)
        std = torch.where(choose_first, std_1, std_2)
        return mean, std

    mean = kernel.loc_model(x).squeeze(-1)
    std = kernel.scale_model(x).squeeze(-1).clamp_min(1e-8)
    return mean, std

def map_reco_with_kernels(df_source, source_kernel, target_kernel, batch_size=200_000):
    ptreco_np = np.array(df_source["pt_reco"].to_numpy(), dtype=np.float32, copy=True)
    pttruth_np = np.array(df_source["pt"].to_numpy(), dtype=np.float32, copy=True)
    eta_np = np.array(df_source["eta"].to_numpy(), dtype=np.float32, copy=True)

    N = len(ptreco_np)
    out = np.empty(N, dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        ptreco = torch.from_numpy(ptreco_np[start:end]).to(device)
        pttruth = torch.from_numpy(pttruth_np[start:end]).to(device)
        eta = torch.from_numpy(eta_np[start:end]).to(device)

        x = torch.stack([pttruth, eta], dim=1)

        with torch.inference_mode():
            source_mean, source_std = kernel_error_mean_std(source_kernel, x)
            target_mean, target_std = kernel_error_mean_std(target_kernel, x)
            mean_delta = target_mean - source_mean
            std_delta = torch.sqrt(target_std * target_std + source_std * source_std).clamp_min(1e-8)
            deltaD = mean_delta + std_delta * torch.randn_like(mean_delta)
            mapped_pt = (ptreco - deltaD).clamp_min(0.0)

        out[start:end] = mapped_pt.detach().cpu().numpy().copy()

        del ptreco, pttruth, eta, x, source_mean, source_std, target_mean, target_std, mean_delta, std_delta, deltaD, mapped_pt

    return out

def main():
    df_sc, sc_folder = load_folder("PtErrSC", fallback_folders=("PtErr",))
    df_oc, oc_folder = load_folder("PtErrOC")
    validate_required_columns(df_sc, sc_folder, ["pt_reco", "pt", "eta"])
    validate_required_columns(df_oc, oc_folder, ["pt_reco"])
    kernel_sc = load_kernel_mixture(CKPT_SC)
    kernel_oc = load_kernel(CKPT_OC)

    recoSC = df_sc["pt_reco"].to_numpy(dtype=np.float32)
    recoOC = df_oc["pt_reco"].to_numpy(dtype=np.float32)
    oc_smeared = map_reco_with_kernels(
        df_sc,
        source_kernel=kernel_sc,
        target_kernel=kernel_oc,
    )

    bins = np.linspace(20000, 200000, 100)

    plt.figure()
    plt.hist(recoOC, bins=bins, histtype="step", density=True, label="OC Reco")
    plt.hist(recoSC, bins=bins, histtype="step", density=True, label="SC Reco")
    plt.hist(oc_smeared, bins=bins, histtype="step", density=True, label="OC Smeared")

    plt.xlabel("pT")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
