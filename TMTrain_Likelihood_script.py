import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit

# Script used to re-calculate binned Likelihood QmisID estimates in the Control Region
# iminuit module is used for likelihood minimisation 

path = "/Users/albaburgosmondejar/Desktop/Dataset4/"

eta_col1, eta_col2 = "l1_eta", "l2_eta"
qpt_col1, qpt_col2 = "l1_q_over_pt", "l2_q_over_pt"
label_col = "opposite_charge" 

PtBinning  = [20, 50, 100, 200, 2600]
EtaBinning = [0, 1.37, 2.0, 2.6]

qpt_edges = np.asarray(PtBinning, dtype=float)
eta_edges = np.asarray(EtaBinning, dtype=float)

NBINS_QPT = len(qpt_edges) - 1  
NBINS_ETA = len(eta_edges) - 1  

EPS = 0
factor_bck = 0.976

# ------------------------
pkl_files = sorted(glob.glob(os.path.join(path, "*.pkl")))
df = pd.concat([pd.read_pickle(f) for f in pkl_files], ignore_index=True)

need = [eta_col1, eta_col2, qpt_col1, qpt_col2, label_col, "m_l1l2", "weight"]
df = df[need].copy()

# ------------------------
mask_Z = (
    df["m_l1l2"].between(81000, 101000) &
    ~df[eta_col1].abs().between(1.37, 1.52) &
    ~df[eta_col2].abs().between(1.37, 1.52)
)

df = df.loc[mask_Z].copy()

# Convert q/pt -> pt(ish) the way you intended (kept your transform)
df[qpt_col1] = (1.0 / df[qpt_col1]).abs() * 1e-3
df[qpt_col2] = (1.0 / df[qpt_col2]).abs() * 1e-3
df[eta_col1] = df[eta_col1].abs()
df[eta_col2] = df[eta_col2].abs()

# ------------------------
# ------------------------
def hist4(df_in: pd.DataFrame) -> np.ndarray:
    data = df_in[[eta_col1, eta_col2, qpt_col1, qpt_col2, "weight"]].to_numpy()
    H, _ = np.histogramdd( data[:, :4], bins=(eta_edges, eta_edges, qpt_edges, qpt_edges),  weights = data[:, 4])
    return H.astype(np.float64)

SS = hist4(df[df[label_col] == 0]) 
OS = hist4(df[df[label_col] == 1]) 

# ------------------------
# NLL
# ------------------------
def nll(par_flat, Array_SS, Array_OS, eps=1e-18):
    SS = np.asarray(Array_SS, dtype=np.float64)
    OS = np.asarray(Array_OS, dtype=np.float64)

    E1, E2, P1, P2 = SS.shape
    assert E1 == E2 and P1 == P2
    E, P = E1, P1

    par = np.asarray(par_flat, dtype=np.float64).reshape(E, P) 

    Psum = par[:, None, :, None] + par[None, :, None, :]  

    ALL = SS + OS

    lam = ALL * Psum

    cell_ll = np.where(SS != 0.0, SS * np.log(lam) - lam, -lam)

    tri = np.triu(np.ones((E, E), dtype=np.float64))
    ll = np.sum(cell_ll * tri[:, :, None, None])

    return -ll

E = NBINS_ETA
P = NBINS_QPT
NPAR = E * P
x0 = np.full(NPAR, 8e-5, dtype=float)

def nll_flat(*theta_flat):
    return nll(np.array(theta_flat, dtype=float), SS, OS, eps=EPS)

m = Minuit(nll_flat, *x0)
for i in range(NPAR):
    m.limits[i] = (0.0, None)

m.migrad(ncall=10_000_000)
m.hesse()
               
eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
qpt_centers = 0.5 * (qpt_edges[:-1] + qpt_edges[1:])

theta_EP = np.array(m.values, dtype=float).reshape(E, P)
err_EP   = np.array([m.errors[i] for i in range(E*P)]).reshape(E, P)

theta = theta_EP.T       # (pt, eta)
theta_err = err_EP.T     # (pt, eta)

pt_idx = [0, 1, 2, 3]

colors = plt.cm.tab10(np.linspace(0, 1, len(pt_idx)))

fig, ax = plt.subplots(figsize=(5, 4))

for c, i in zip(colors, pt_idx):
    for j in range(E):
        ax.hlines(
            y=theta[i, j],
            xmin=eta_edges[j],
            xmax=eta_edges[j + 1],
            color=c,
            linewidth=2,
            label = rf"$p_T \in [{PtBinning[i]}, {PtBinning[i+1]}]$ GeV" if j == 0 else None
        )

        ax.vlines(
            x=eta_centers[j],
            ymin=theta[i, j] - theta_err[i, j],
            ymax=theta[i, j] + theta_err[i, j],
            color=c,
            linewidth=1
        )

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=len(pt_idx),
    frameon=False
)

ax.set_xlabel(r"$|\eta|$")
ax.set_ylabel("qmisid")
ax.set_yscale("log")
ax.set_xlim(eta_edges[0], eta_edges[-1])
ax.grid(True, which="both", alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

ALL4 = SS + OS  

counts_eta_pt = ALL4.sum(axis=(1, 3))     
counts_pt_eta = counts_eta_pt.T          

fig, ax = plt.subplots(figsize=(4, 4))
pc = ax.pcolormesh(eta_edges, qpt_edges, counts_pt_eta, shading="auto")

ax.set_xlabel("eta (bin edges)")
ax.set_ylabel("pt (bin edges)")
ax.set_yscale("log")
fig.colorbar(pc, ax=ax, label="N events (SS+OS)")

ax.set_xticks(eta_edges)
ax.set_xticklabels([f"{v:.2g}" for v in eta_edges])
ax.set_yticks(qpt_edges)
ax.set_yticklabels([f"{v:.2g}" for v in qpt_edges])

eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
qpt_centers = 0.5 * (qpt_edges[:-1] + qpt_edges[1:])

for i in range(NBINS_QPT):     
    for j in range(NBINS_ETA): 
        ax.text(
            eta_centers[j],
            qpt_centers[i],
            f"{int(counts_pt_eta[i, j])}",
            ha="center", va="center",
            fontsize=8
        )

plt.tight_layout()
plt.show()

for i in range(theta.shape[0]):
    for j in range(theta.shape[1]):
        print(f"theta[{i},{j}] = {theta[i, j]:.6g}")