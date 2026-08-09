# Closure test January 

import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 20,
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "legend.fontsize": 18,
})

from TMTrain import KappaLightning as KappaMC
from TMTrain import KappaLightning as KappaTruth
from iwpc.utils import latest_ckpt

ckpt_path = latest_ckpt("dielectron_logs/kappa/version_42/")
module: KappaMC = KappaMC.load_from_checkpoint(ckpt_path, map_location="cpu")
module.eval() 

ckpt_path2 = latest_ckpt("dielectron_logs/kappa/version_42/")
module2: KappaTruth = KappaTruth.load_from_checkpoint(ckpt_path2, map_location="cpu")
module2.eval() 

def convert_to_prob(kappa: np.ndarray) -> np.ndarray:
    return 1-(1 / (1 + np.exp(-kappa)))

def convert_to_prob2(kappa: np.ndarray) -> np.ndarray:
    return (1 / (1 + np.exp(-kappa)))

@torch.no_grad()
def evaluate_for_visualiser(z: np.ndarray, model: torch.nn.Module = module) -> dict:
    z_proc = np.stack([
        z[:, 0] / z[:, 1],
        z[:, 2],
    ], axis=1).astype(np.float32)
    z_tensor = torch.from_numpy(z_proc)
    output = model(z_tensor) 
    kappa = output[:, 0].detach().cpu().numpy()
    prob = convert_to_prob(kappa)             

    return prob

@torch.no_grad()
def evaluate_for_visualiser2(z: np.ndarray, model: torch.nn.Module = module2) -> dict:
    z_proc = np.stack([
        z[:, 0] / z[:, 1],
        z[:, 2],
    ], axis=1).astype(np.float32)
    z_tensor = torch.from_numpy(z_proc)
    output = model(z_tensor) 
    kappa = output[:, 0].detach().cpu().numpy()
    prob = convert_to_prob2(kappa)             
    return prob

pt_vals = np.linspace(5e3, 260000, 500)   
eta_vals = np.linspace(0.0, 2.6, 500)  

ETA, PT = np.meshgrid(eta_vals, pt_vals)
charge = -1.0 

z_input = np.column_stack([
    np.full(PT.size, charge),
    PT.ravel(),
    ETA.ravel()
]).astype(np.float32)

p_grid = evaluate_for_visualiser(z_input, module)
p_grid = p_grid.reshape(PT.shape)

### Closure test 

import uproot
import awkward as ak

files = [
   "/Users/albaburgosmondejar/Desktop/Input2/hhml_v2_2lSC_QMisID_Vjets_Zee.root",
]

arrays = [uproot.open(f)["qmisid_cr"].arrays() for f in files]
branches = ak.concatenate(arrays, axis=0)
n = len(branches)
branches = branches[int(0.8 * n):]

PtBinning  = [20*1000, 50*1000, 100*1000, 200*1000, 2600*1000] 
EtaBinning = [0, 1.37, 2.0, 2.6]

mask= (branches["m_l1l2"] > 81e3) & (branches["m_l1l2"] < 101e3)
maskw =branches["weight"] > 0
#mask = (branches["m_l1l2"] < 81000) | (branches["m_l1l2"] > 101000) 
mask1 = (branches["same_charge"] == 1) & mask & maskw
mask0 = (branches["same_charge"] == 0) & mask & maskw

def grid_prob(pt, eta):

    pt = np.asarray(pt, float)
    eta = np.abs(np.asarray(eta, float)) 

    pt_c  = np.clip(pt,  pt_vals[0],  pt_vals[-1])
    eta_c = np.clip(eta, eta_vals[0], eta_vals[-1])

    i = np.searchsorted(pt_vals,  pt_c,  side="right") - 1
    j = np.searchsorted(eta_vals, eta_c, side="right") - 1
    i = np.clip(i, 0, len(pt_vals) - 2)
    j = np.clip(j, 0, len(eta_vals) - 2)

    t = (pt_c  - pt_vals[i])  / np.clip(pt_vals[i+1]  - pt_vals[i],  1e-12, None)
    u = (eta_c - eta_vals[j]) / np.clip(eta_vals[j+1] - eta_vals[j], 1e-12, None)

    p00 = p_grid[i,   j  ]
    p10 = p_grid[i+1, j  ]
    p01 = p_grid[i,   j+1]
    p11 = p_grid[i+1, j+1]

    p0 = p00 * (1 - t) + p10 * t
    p1 = p01 * (1 - t) + p11 * t
    p  = p0  * (1 - u) + p1  * u

    return np.clip(p, 1e-8, 1 - 1e-8)

def w_oc_to_sc(p1, p2):
    Psc = p1*(1-p2) + (1-p1)*p2
    Poc = (1-p1)*(1-p2) + p1*p2
    return Psc / np.clip(Poc, 1e-12, None)


# Likelihood Calculation

theta = np.zeros((4, 3), dtype=float)
theta[0,0] = 0.000133191
theta[0,1] = 0.00073846
theta[0,2] = 0.00121403
theta[1,0] = 0.000231969
theta[1,1] = 0.00131285
theta[1,2] = 0.00233867
theta[2,0] = 0.0016104
theta[2,1] = 0.00767679
theta[2,2] = 0.0135268
theta[3,0] = 0.00537655
theta[3,1] = 0.0286459
theta[3,2] = 0.0448349


def grid_probL(pt, eta):
    pt  = np.asarray(pt, float)
    eta = np.abs(np.asarray(eta, float))

    pt_c  = np.clip(pt,  PtBinning[0],  PtBinning[-1] )
    eta_c = np.clip(eta, EtaBinning[0], EtaBinning[-1] )

    i = np.searchsorted(PtBinning,  pt_c) - 1
    j = np.searchsorted(EtaBinning, eta_c) - 1

    i = np.clip(i, 0, len(PtBinning) - 2)
    j = np.clip(j, 0, len(EtaBinning) - 2)

    return theta[i, j]

### Figures
weights = np.asarray(branches["weight"], float)
p1 = grid_prob(branches["l1_pt"], branches["l1_eta"])
p2 = grid_prob(branches["l2_pt"], branches["l2_eta"])
w  = np.asarray(w_oc_to_sc(p1, p2) * weights)
p1L = grid_probL(branches["l1_pt"], branches["l1_eta"])
p2L = grid_probL(branches["l2_pt"], branches["l2_eta"])
wL  = np.asarray(w_oc_to_sc(p1L, p2L) * weights)
mask0 = np.asarray(mask0)
mask1 = np.asarray(mask1)


var_specs = [
    ("l1_pt",  lambda b: np.asarray(b["l1_pt"], float) / 1e3, r"$p_T^{\ell_1}$ [GeV]", 20, 200),
    ("l2_pt",  lambda b: np.asarray(b["l2_pt"], float) / 1e3, r"$p_T^{\ell_2}$ [GeV]", 20, 200),
    ("l1_eta", lambda b: np.abs(np.asarray(b["l1_eta"], float)), r"$|\eta^{\ell_1}|$", 0.01, 2.6),
    ("l2_eta", lambda b: np.abs(np.asarray(b["l2_eta"], float)), r"$|\eta^{\ell_2}|$", 0.01, 2.6),
    ("m_l1l2", lambda b: np.asarray(b["m_l1l2"], float) / 1e3, r"$m_{\ell_1\ell_2}$ [GeV]", 0, 200),
]

nbins = 30

def binned_counts_and_errors(x, bins, xmin, xmax, weights=None):
    counts, edges = np.histogram(x, bins=bins, range=(xmin, xmax), weights=weights)
    errs = np.sqrt(np.abs(counts))
    centers = 0.5 * (edges[1:] + edges[:-1])
    widths  = (edges[1:] - edges[:-1])
    return centers, counts, errs, widths


def plot_panel(x, xlabel, xmin, xmax, title=None):
    bins = np.linspace(xmin, xmax, nbins + 1)

    x_sc = np.asarray(x[mask1], float)
    x_oc = np.asarray(x[mask0], float)

    c_oc, y_oc, e_oc, wbin = binned_counts_and_errors(
        x_oc, bins=bins, xmin=xmin, xmax=xmax, weights=w[mask0]
    )
    c_sc, y_sc, e_sc, _ = binned_counts_and_errors(
        x_sc, bins=bins, xmin=xmin, xmax=xmax, weights=weights[mask1]
    )
    c_2, y_2, e_2, _ = binned_counts_and_errors(
        x_oc, bins=bins, xmin=xmin, xmax=xmax, weights=weights[mask0]
    )
    c_L, y_L, e_L, _ = binned_counts_and_errors(
        x_oc, bins=bins, xmin=xmin, xmax=xmax, weights=wL[mask0]
    )

    xerr = 0.5 * wbin

    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_cmp = fig.add_subplot(gs[1, 0], sharex=ax_main)

    ax_main.errorbar(c_sc, y_sc, yerr=e_sc, xerr=xerr, fmt="none", lw=1, capsize=2,
                     color="black", label=r"$\mathrm{SC}$")
    ax_main.errorbar(c_2, y_2, yerr=e_2, xerr=xerr, fmt="none", lw=1, capsize=2,
                     color="tab:purple", label=r"$\mathrm{OC}$")
    ax_main.errorbar(c_L, y_L, yerr=e_L, xerr=xerr, fmt="o", ms=3.5, lw=1, capsize=2,
                     color="tab:red", label=r"$\mathrm{OC~weighted~L}$")
    ax_main.errorbar(c_oc, y_oc, yerr=e_oc, xerr=xerr, fmt="o", ms=3.5, lw=1, capsize=2,
                     color="tab:orange", label=r"$\mathrm{OC~weighted~NN}$")

    ax_main.set_yscale("log")
    ax_main.set_ylabel("Events")
    if title:
        ax_main.set_title(title)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    rel     = np.full_like(y_sc, np.nan, dtype=float)
    rel_err = np.full_like(y_sc, np.nan, dtype=float)
    rel_L   = np.full_like(y_sc, np.nan, dtype=float)
    rel_err_L = np.full_like(y_sc, np.nan, dtype=float)

    m = y_sc > 0
    rel[m] = (y_oc[m] - y_sc[m]) / y_sc[m]
    rel_err[m] = np.sqrt((e_oc[m] / y_sc[m])**2 + ((y_oc[m] * e_sc[m]) / (y_sc[m]**2))**2)

    rel_L[m] = (y_L[m] - y_sc[m]) / y_sc[m]
    rel_err_L[m] = np.sqrt((e_L[m] / y_sc[m])**2 + ((y_L[m] * e_sc[m]) / (y_sc[m]**2))**2)

    ax_cmp.axhline(0.0, color="0.3", lw=1)
    ax_cmp.errorbar(c_sc, rel, yerr=rel_err, xerr=xerr, fmt="o", ms=2.5, lw=1, capsize=2,
                    color="tab:orange")
    ax_cmp.errorbar(c_sc, rel_L, yerr=rel_err_L, xerr=xerr, fmt="o", ms=2.5, lw=1, capsize=2,
                    color="tab:red")
    ax_cmp.set_ylabel(r"$(\mathrm{OC_{weighted}}-\mathrm{SC})/\mathrm{SC}$")
    ax_cmp.set_xlabel(xlabel)
    ax_cmp.set_ylim(-1, 1)

    ax_main.legend()
    return fig

for _, getter, xlabel, xmin, xmax in var_specs:
    x = getter(branches)
    plot_panel(x, xlabel, xmin, xmax, title=xlabel)

plt.show()
