from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from gaussian_kernel_single import make_kernel
from gaussian_kernel_variable_alpha import make_mixture_kernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_OC = "kernel_OC/gaussian/version_66/checkpoints/epoch=8-step=13734.ckpt"
CKPT_SC = "kernel_SC/gaussian_mixture/version_5/checkpoints/epoch=70-step=1349.ckpt"

def load_folder(folder: str) -> pd.DataFrame:
    files = sorted(Path(folder).glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {folder}")
    dfs = [pd.read_pickle(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

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


df_sc = load_folder("PtErrSC")
df_oc = load_folder("PtErrOC")

kernelSC = load_kernel_mixture(CKPT_SC)
kernelOC = load_kernel(CKPT_OC)

def smear_df(df, kernel, name="", batch_size=200_000):
    # Ensure writable arrays for torch.from_numpy.
    ptreco_np = np.array(df["pt_reco"].to_numpy(), dtype=np.float32, copy=True)
    pttruth_np = np.array(df["pt"].to_numpy(), dtype=np.float32, copy=True)
    eta_np    = np.array(df["eta"].to_numpy(), dtype=np.float32, copy=True)

    N = len(ptreco_np)

    smeared_list = []
    reco_list = []
    truth_list = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        ptreco = torch.from_numpy(ptreco_np[start:end]).to(device)
        pttruth = torch.from_numpy(pttruth_np[start:end]).to(device)
        eta = torch.from_numpy(eta_np[start:end]).to(device)

        x = torch.stack([pttruth, eta], dim=1)

        with torch.inference_mode():
            err = kernel.draw(x)

            if err.ndim == 2:
                err = err[:, 0]

        ptsmeared = (pttruth - err).clamp_min(0.0)

        reco_list.append(ptreco.cpu().numpy())
        smeared_list.append(ptsmeared.cpu().numpy())
        truth_list.append(pttruth.cpu().numpy())

        del ptreco, eta, x, err, ptsmeared, pttruth

        if device.type == "cuda":
            torch.cuda.empty_cache()

    
    print(f"{name} processed in batches.")

    return (
        np.concatenate(reco_list),
        np.concatenate(smeared_list),
        np.concatenate(truth_list),
    )

ptrecoSC, pt_smearedSC, pttruthSC = smear_df(df_sc, kernelSC, "SC")
ptrecoOC, pt_smearedOC, pttruthOC = smear_df(df_oc, kernelOC, "OC")

bins = np.linspace(0, 200000, 100)

plt.figure()
plt.hist(ptrecoSC,      bins=bins, histtype="step", density=True, label="ptrecoSC")
plt.hist(pt_smearedSC,  bins=bins, histtype="step", density=True, label="pt_smearedSC")
plt.hist(ptrecoOC,      bins=bins, histtype="step", density=True, label="ptrecoOC")
plt.hist(pt_smearedOC,  bins=bins, histtype="step", density=True, label="pt_smearedOC")
plt.hist(pttruthSC,      bins=bins, histtype="step", density=True, label="pttruthSC")
plt.hist(pttruthOC,  bins=bins, histtype="step", density=True, label="pttruthOC")

plt.xlabel("pT")
plt.ylabel("density")
plt.legend()
plt.tight_layout()
plt.show()
