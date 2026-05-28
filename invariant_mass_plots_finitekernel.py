# Invariant Mass plots for DielectronUnlabelled.py
# Version for FiniteKernel interface with mixture gaussian pt smearing
# This version still needs to be trained to completion 

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from iwpc.symmetries.symmetrized_model import SymmetrizedModel
from iwpc.symmetries.finite_group_action import FiniteGroupAction
from iwpc.symmetries.lambda_action import LambdaAction
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.encodings.nope_encoding import NopeEncoding
from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.kernels.branching_kernel import BranchingKernel
from iwpc.learn_dist.kernels.finite_kernel import FiniteKernel
from iwpc.learn_dist.kernels.constant_kernel import ConstantKernel
from iwpc.learn_dist.kernels.permutation_kernel import PermutationKernel
from iwpc.models.utils import basic_model_factory
from iwpc.models.layers import ConstantScaleLayer
from iwpc.learn_dist.kernels.partially_exact_unlabelled_kernel_trainer import PartiallyExactUnLabelledKernelTrainer
from InvariantMass import CustomDielectronEncoding
from torch.nn.functional import logsigmoid
from iwpc.learn_dist.fdivergence_minimization.fdivergence_minimizing_kernel_trainer import FDivergenceMinimizingKernelTrainer
from iwpc.divergences import DifferentiableFDivergence, KLDivergence, JensenShannonDivergence

DEVICE = torch.device("cpu")
DATA_DIR = "/Users/albaburgosmondejar/Desktop/BenchmarkMC"

# ── rebuild model (DielectronUnlabelled.py) ────────────────────────
single_electron_encoding = TrivialEncoding(3)
shift_scale = np.log(1e2)
shift_loc = -1.2e3

single_electron_flip_kernel = FiniteKernel(3, single_electron_encoding, init_log_probs=[np.log(0.998), np.log(0.001), np.log(0.001)])
event_flip_kernel = single_electron_flip_kernel + single_electron_flip_kernel
exactly_one_flip_kernel = event_flip_kernel.cut(lambda x: (x.sum(dim=-1) > 0) and (x[0] == 0 or x[1] == 0))

single_electron_no_flip_kernel = ConstantKernel([0], single_electron_encoding)
single_electron_brehm_kernel = GaussianKernel(
    single_electron_encoding,
    loc_model=basic_model_factory(single_electron_encoding, 1, final_layers=[ConstantScaleLayer(shift_loc, np.exp(shift_scale))]),
    scale_model=basic_model_factory(single_electron_encoding, ExponentialEncoding(1), final_layers=[ConstantScaleLayer(shift_scale)]),
)
single_electron_res_kernel = GaussianKernel(
    single_electron_encoding,
    loc_model=basic_model_factory(single_electron_encoding, 1, final_layers=[ConstantScaleLayer(shift_loc, np.exp(shift_scale))]),
    scale_model=basic_model_factory(single_electron_encoding, ExponentialEncoding(1), final_layers=[ConstantScaleLayer(shift_scale)]),
)
single_electron_eta_kernel = ConstantKernel([0], TrivialEncoding(4))
single_electron_phi_kernel = ConstantKernel([0], TrivialEncoding(4))
single_electron_pt_kernel = BranchingKernel.condition_on([
    single_electron_no_flip_kernel,
    single_electron_res_kernel,
    single_electron_brehm_kernel,
], single_electron_flip_kernel) & single_electron_eta_kernel & single_electron_phi_kernel 

dielectron_smear_kernel = single_electron_pt_kernel + single_electron_pt_kernel

reordered_dielectron_smear_kernel = PermutationKernel(
    dielectron_smear_kernel,
    cond_permutation=[0, 2, 3, 4, 1,5, 6, 7],
)
# print(reordered_dielectron_smear_kernel.cond_dimension, event_flip_kernel.sample_dimension, event_flip_kernel.cond_dimension)
# full_kernel = reordered_dielectron_smear_kernel | event_flip_kernel.cut(lambda x: (x.sum(dim=-1) > 0) and (x[0] == 0 or x[1] == 0))
# print(full_kernel.draw(torch.zeros(10, 6)))

# model = basic_model_factory(TrivialEncoding(2) & NopeEncoding(1)& TrivialEncoding(2) & NopeEncoding(3))
model = basic_model_factory(CustomDielectronEncoding(TrivialEncoding(2) & NopeEncoding(1)& TrivialEncoding(2) & NopeEncoding(3)))
swap_element = LambdaAction(
    input_dim=8,
    output_dim=1,
    input_fn=lambda x: torch.cat([x[:, 3:6], x[:, 0:3], x[:, 6:]], dim=-1),
)
group_action = FiniteGroupAction([swap_element], input_dim=8, output_dim=1)
symmetrized_model = SymmetrizedModel(group_action, model)
unlabelled = FDivergenceMinimizingKernelTrainer(exact_kernel=exactly_one_flip_kernel, sampled_kernel= reordered_dielectron_smear_kernel, log_p_over_q_model= symmetrized_model, divergence=KLDivergence())

# ── load latest checkpoint ────────────────────────────────────────────────────
ckpt_candidates = sorted(
    Path("div_logs/di_electron/version_12/").glob("checkpoints/*.ckpt"),
    key=lambda p: p.stat().st_mtime,
)
if not ckpt_candidates:
    raise FileNotFoundError("No checkpoint found under grad_logs/di_electron/")
CKPT_PATH = ckpt_candidates[-1]
print(f"Loading checkpoint: {CKPT_PATH}")

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
missing, unexpected = unlabelled.load_state_dict(ckpt["state_dict"], strict=False)
if missing:
    print(f"WARNING: {len(missing)} keys not loaded: {missing[:3]}...")
unlabelled.eval()

# ── helpers ───────────────────────────────────────────────────────────────────
def load_data(max_files=None):
    files = sorted(Path(DATA_DIR).glob("file_*.pkl"))
    if max_files is not None:
        files = files[:max_files]
    return pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)


def inv_mass(pt1, eta1, phi1, pt2, eta2, phi2):
    return np.sqrt(np.maximum(2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2)), 0))


@torch.no_grad()
def sample_smeared_oc(base_np, weight_np, batch_size=50_000):
    all_masses, all_weights, smeared_pt1, smeared_pt2 = [], [], [], []

    base = torch.tensor(base_np, dtype=torch.float32, device=DEVICE)
    weights = torch.tensor(weight_np, dtype=torch.float32, device=DEVICE)

    for start in tqdm(range(0, len(base), batch_size)):
        b = base[start:start + batch_size]
        w = weights[start:start + batch_size]

        w0 = 0
        q = None
        for sampled_kernel_cond, exact_outcome_log_prob in unlabelled.sampled_kernel_cond_iter(b):
            samples, log_prob = unlabelled.sampled_kernel.draw_with_log_prob(sampled_kernel_cond)
            q = torch.cat([b + samples, torch.zeros(len(samples), 2, device=DEVICE)], dim=-1)
            log_p_over_q = unlabelled.calculate_log_p_over_q(q)
            w0 = w0 + w * exact_outcome_log_prob.exp() * log_p_over_q.exp()

        w_np = w.cpu().numpy()
        q_np = q.cpu().numpy()
        all_masses.append(inv_mass(q_np[:, 0], q_np[:, 1], q_np[:, 2],
                                   q_np[:, 3], q_np[:, 4], q_np[:, 5]))
        all_weights.append(w_np)
        smeared_pt1.append(q_np[:, 0])
        smeared_pt2.append(q_np[:, 3])

    return (np.concatenate(all_masses), np.concatenate(all_weights),
            np.concatenate(smeared_pt1), np.concatenate(smeared_pt2))

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(max_files=2)
    oc = df[df['opposite_charge'] == 1]
    sc = df[df['same_charge'] == 1]

    oc_weight = oc['weight'].values
    sc_weight = sc['weight'].values

    base_np = oc[['l1_pt', 'l1_eta', 'l1_phi', 'l2_pt', 'l2_eta', 'l2_phi']].values.astype(np.float32)
    smeared_mass, smeared_weight, smeared_pt1, smeared_pt2 = sample_smeared_oc(base_np, oc_weight.astype(np.float32))
    # smeared_pt = np.concatenate([smeared_pt1, smeared_pt2])

    mass_bins = np.linspace(60_000, 120_000, 80)
    pt_bins = np.linspace(0,1.5e5, 80)
    kw = dict(histtype='step', linewidth=1.5, density=True)

    fig, (ax_mass, ax_pt1, ax_pt2) = plt.subplots(1, 3, figsize=(21, 5))

    ax_mass.hist(oc['m_l1l2'].values, bins=mass_bins, weights=oc_weight,      label='Data OC',    color='tab:blue',   **kw)
    ax_mass.hist(smeared_mass,         bins=mass_bins, weights=smeared_weight, label='Simulated+Reweighted OC', color='tab:green',  **kw)
    ax_mass.hist(sc['m_l1l2'].values,  bins=mass_bins, weights=sc_weight,      label='Data SC',    color='tab:orange', **kw)
    ax_mass.hist(smeared_mass,  bins=mass_bins, weights=oc_weight,      label='Simulated',    color='tab:red', **kw)
    ax_mass.set_xlabel('Invariant mass [MeV]')
    ax_mass.set_ylabel('Density')
    ax_mass.set_title('Dielectron invariant mass — Z region [60–120 GeV]')
    ax_mass.legend(fontsize=8)

    ax_pt1.hist(base_np[:, 0], bins=pt_bins, weights=oc_weight,   label='Data OC',    color='tab:blue',   **kw)
    ax_pt1.hist(smeared_pt1,     bins=pt_bins, weights=smeared_weight, label='Simulated+Reweighted OC', color='tab:green',  **kw)
    ax_pt1.hist(sc['l1_pt'].values, bins=pt_bins, weights=sc_weight,   label='Data SC',    color='tab:orange', **kw)
    ax_pt1.hist(smeared_pt1, bins=pt_bins, weights=oc_weight,   label='Simulated',    color='tab:red', **kw)
    ax_pt1.set_xlabel('Lepton $p_{T_1}$ [MeV]')
    ax_pt1.set_ylabel('Density')
    ax_pt1.set_title('Lepton $p_{T_1}$ distribution')
    ax_pt1.legend(fontsize=8)

    ax_pt2.hist(base_np[:, 3], bins=pt_bins, weights=oc_weight,   label='Data OC',    color='tab:blue',   **kw)
    ax_pt2.hist(smeared_pt2,     bins=pt_bins, weights=smeared_weight, label='Simulated+Reweighted OC', color='tab:green',  **kw)
    ax_pt2.hist(sc['l2_pt'].values, bins=pt_bins, weights=sc_weight,   label='Data SC',    color='tab:orange', **kw)
    ax_pt2.hist(smeared_pt2, bins=pt_bins, weights=oc_weight,   label='Simulated',    color='tab:red', **kw)
    ax_pt2.set_xlabel('Lepton $p_{T_2}$ [MeV]')
    ax_pt2.set_ylabel('Density')
    ax_pt2.set_title('Lepton $p_{T_2}$ distribution')
    ax_pt2.legend(fontsize=8)

    plt.tight_layout()
    out = Path("invariant_mass.png")
    plt.savefig(out, dpi=150)
    plt.show()
