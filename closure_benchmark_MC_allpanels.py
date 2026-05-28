# Closure test with BenchmarkMC, 5 panels joint

import glob
import pandas as pd
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
    "legend.fontsize": 14,
})

from iwpc.symmetries.symmetrized_model import SymmetrizedModel
from iwpc.symmetries.finite_group_action import FiniteGroupAction
from iwpc.symmetries.lambda_action import LambdaAction
import numpy as np
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.encodings.nope_encoding import NopeEncoding
from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.kernels.branching_kernel import BranchingKernel
from iwpc.learn_dist.kernels.finite_kernel import FiniteKernel
from iwpc.learn_dist.kernels.finite_cut_kernel import FiniteCutKernel
from iwpc.learn_dist.kernels.mixture_kernel import MixtureKernel
from iwpc.learn_dist.kernels.constant_kernel import ConstantKernel
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.learn_dist.kernels.unlabelled_kernel_trainer import UnLabelledKernelTrainer
from iwpc.models.utils import basic_model_factory
from iwpc.models.layers import ConstantScaleLayer
from iwpc.learn_dist.fdivergence_minimization.fdivergence_minimizing_kernel_trainer import FDivergenceMinimizingKernelTrainer
from iwpc.learn_dist.kernels.permutation_kernel import PermutationKernel
from InvariantMass import CustomDielectronEncoding
from iwpc.divergences import DifferentiableFDivergence, KLDivergence, JensenShannonDivergence
from iwpc.learn_dist.kernels.finite_kernel import FiniteKernelInterface
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
import matplotlib.pyplot as plt
from DielectronUnlabelled import DielectronKernelTrainer

single_electron_encoding = TrivialEncoding(3)
shift_scale = np.log(1e3)
shift_loc   = -1.2e3

single_electron_flip_kernel = FiniteKernel(
    2, single_electron_encoding,
    init_log_probs=[np.log(0.998), np.log(0.002)],
)
event_flip_kernel       = single_electron_flip_kernel + single_electron_flip_kernel
exactly_one_flip_kernel = FiniteCutKernel(
    event_flip_kernel,
    lambda x: (x.sum(dim=-1) > 0) and (x[0] == 0 or x[1] == 0),
)

single_electron_no_flip_kernel = ConstantKernel([0], single_electron_encoding)
single_electron_res_kernel = GaussianKernel(
    single_electron_encoding,
    loc_model=basic_model_factory(
        single_electron_encoding, 1,
        final_layers=[ConstantScaleLayer(shift_loc, np.exp(shift_scale))],
    ),
    scale_model=basic_model_factory(
        single_electron_encoding, ExponentialEncoding(1),
        final_layers=[ConstantScaleLayer(shift_scale)],
    ),
)
single_electron_eta_kernel = ConstantKernel([0], TrivialEncoding(4))
single_electron_phi_kernel = ConstantKernel([0], TrivialEncoding(4))
single_electron_pt_kernel  = BranchingKernel.condition_on(
    [single_electron_no_flip_kernel, single_electron_res_kernel],
    single_electron_flip_kernel,
) & single_electron_eta_kernel & single_electron_phi_kernel

dielectron_smear_kernel = single_electron_pt_kernel + single_electron_pt_kernel
reordered_dielectron_smear_kernel = PermutationKernel(
    dielectron_smear_kernel,
    cond_permutation=[0, 2, 3, 4, 1, 5, 6, 7],
)

model = basic_model_factory(
    CustomDielectronEncoding(
        TrivialEncoding(2) & NopeEncoding(1) & TrivialEncoding(2) & NopeEncoding(3)
    )
)
swap_element = LambdaAction(
    input_dim=8,
    output_dim=1,
    input_fn=lambda x: torch.cat([x[:, 3:6], x[:, 0:3], x[:, 6:]], dim=-1),
)
symmetrized_model = SymmetrizedModel(
    FiniteGroupAction([swap_element], input_dim=8, output_dim=1), model
)

trainer = DielectronKernelTrainer.load_from_checkpoint(
    "iwpc_logs/di_electron/version_66/checkpoints/epoch=2-step=12822.ckpt",
    exact_kernel=exactly_one_flip_kernel,
    sampled_kernel=reordered_dielectron_smear_kernel,
    log_p_over_q_model=symmetrized_model,
    start_kernel_train_epoch=0,
    start_discriminator_train_epoch=0,
    kernel_opt_lr=1e-4,
    divergence=JensenShannonDivergence(),
    map_location="cpu",
    target_cut_pass_prob=torch.tensor(0.001),
)
trainer.eval()


# ── Likelihood baseline ────────────────────────────────────────────────────────

PtBinning  = [20e3, 50e3, 100e3, 200e3, 2600e3]
EtaBinning = [0, 1.37, 2.0, 2.6]
theta = np.array([
    [0.000133191, 0.00073846,  0.00121403],
    [0.000231969, 0.00131285,  0.00233867],
    [0.0016104,   0.00767679,  0.0135268 ],
    [0.00537655,  0.0286459,   0.0448349 ],
])

def grid_prob(pt, eta):
    pt  = np.clip(np.asarray(pt,  float), PtBinning[0],  PtBinning[-1])
    eta = np.clip(np.abs(np.asarray(eta, float)), EtaBinning[0], EtaBinning[-1])
    i = np.clip(np.searchsorted(PtBinning,  pt)  - 1, 0, len(PtBinning)  - 2)
    j = np.clip(np.searchsorted(EtaBinning, eta) - 1, 0, len(EtaBinning) - 2)
    return theta[i, j]

def w_oc_to_sc(p1, p2):
    Psc = p1 * (1 - p2) + (1 - p1) * p2
    Poc = (1 - p1) * (1 - p2) + p1 * p2
    return Psc / np.clip(Poc, 1e-12, None)

# ── Kernel simulation ──────────────────────────────────────────────────────────


dm = PandasDirDataModule(
        dataset_dir="/Users/albaburgosmondejar/Desktop/BenchmarkMC",
        feature_spec=[
            ["l1_pt", "l1_eta", "l1_phi", "l2_pt", "l2_eta", "l2_phi"],  # base: OC kinematics (6D), fed into exact_kernel
            ["l1_pt", "l1_eta","l1_phi", "l2_pt", "l2_eta", "l2_phi", "b", "b"], # data: SC observables (8D), fed into discriminator
            'opposite_charge',
        ],
        weight_col="weight",
        split=0.3,
        dataloader_kwargs={"num_workers": 0, "batch_size": 2**20},
        # use_in_memory_dataset=True,
    )

def plotting(batch):
                  
    base_samples, _, labels, weights = batch
    q_base = base_samples[labels==1]
    sc = base_samples[labels==0].cpu().numpy()
    sc_weights = weights[labels==0].cpu().numpy()
    oc = base_samples[labels==1].cpu().numpy()
    oc_weights = weights[labels==1].cpu().numpy()

    simulated_all   = []
    model_weights_all = []


    with torch.no_grad():
        exact_outcome_log_prob_iter, cut_pass_log_prob = trainer.exact_kernel.outcome_with_log_prob_iter_and_cut_pass_log_prob(q_base)
        exact_outcome_log_prob_iter = list(exact_outcome_log_prob_iter)
        norm2 = (oc_weights.mean()) /(sc_weights.mean())
        flip_prob = cut_pass_log_prob.exp().cpu().numpy() * oc_weights 
        
        first_outcome,  first_log_prob  = exact_outcome_log_prob_iter[0]
        second_outcome, second_log_prob = exact_outcome_log_prob_iter[1]
        use_first = (first_log_prob > second_log_prob).unsqueeze(-1)  # (N, 1)
        outcome_cond = torch.where(
            use_first,
            first_outcome.unsqueeze(0).expand(q_base.shape[0], -1),
            second_outcome.unsqueeze(0).expand(q_base.shape[0], -1),
        ) 
        smear = (q_base + trainer.sampled_kernel.draw(
            torch.cat([outcome_cond, q_base], dim=1)
        )).cpu().numpy()

        for exact_outcome, exact_log_prob in exact_outcome_log_prob_iter:
            repeated_exact_outcome = exact_outcome.repeat((q_base.shape[0], 1))
            sampled_kernel_cond = torch.concat([repeated_exact_outcome, q_base], dim=1)
            samples = trainer.sampled_kernel.draw(sampled_kernel_cond)
            simulation = q_base + samples
            simulated_all.append(simulation.cpu().numpy())
            model_weights = trainer.log_p_over_q_model(torch.concat([ simulation, repeated_exact_outcome], dim=1)).exp().cpu().numpy().squeeze(-1)
            model_weights_all.append(model_weights*oc_weights)

    simulated = np.concatenate(simulated_all)
    model_weight = np.concatenate(model_weights_all)
    norm = np.sum(sc_weights) / np.sum(model_weight)
    model_weight = model_weight * norm

    p1L = grid_prob(oc[:, 0], oc[:, 1])
    p2L = grid_prob(oc[:, 3], oc[:, 4])
    wL  = w_oc_to_sc(p1L, p2L) * oc_weights

    def inv_mass(x):
        pt1, eta1, phi1, pt2, eta2, phi2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        return np.sqrt(np.maximum(2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2)), 0))

    def phist(ax, x, bins, weights=None, **kwargs):
        color = kwargs.get('color')
        h, edges = np.histogram(x, bins=bins, weights=weights)
        n, _ = np.histogram(x, bins=bins)
        err = np.where(n > 0, np.sqrt(np.abs(h)), 0.0)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.hist(x, bins=bins, weights=weights, **kwargs)
        ax.errorbar(centers, h, yerr=err, fmt='none', color=color, capsize=2, elinewidth=0.8)

    kw = dict(histtype='step', linewidth=1.5)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for ax, oc_vals, smear, sc_vals, sim_vals, bins, xlabel in [
            (axes[0], oc[:, 0], smear[:,0], sc[:, 0], simulated[:, 0], np.linspace(25e3,    200e3, 30), 'l1_pt [MeV]'),
            (axes[1], oc[:, 1], smear[:,1], sc[:, 1], simulated[:, 1], np.linspace(-2.5, 2.5,   30), 'l1_eta'),
            (axes[2], oc[:, 3], smear[:,3], sc[:, 3], simulated[:, 3], np.linspace(25e3, 200e3, 30), 'l2_pt [MeV]'),
            (axes[3], oc[:, 4], smear[:,4], sc[:, 4], simulated[:, 4], np.linspace(-2.5, 2.5, 30), 'l2_eta'),
            (axes[4], inv_mass(oc), inv_mass(smear), inv_mass(sc), inv_mass(simulated), np.linspace(81e3, 101e3, 30), 'm_l1l2 [MeV]'),
        ]:
    
            phist(ax, smear,   bins=bins, weights=flip_prob, label='OC smeared × P(flip) × w', color='tab:green',  **kw)
            # phist(ax, sim_vals, bins=bins, weights=model_weight,          label='Simulated × p/q × w',      color='tab:blue',   **kw)
            phist(ax, oc_vals, bins=bins, weights=wL,                     label='OC × likelihood w',         color='tab:purple', **kw)
            phist(ax, sc_vals, bins=bins, weights=sc_weights,             label='Real SC',                   color='tab:orange', **kw)
            # phist(ax, oc_vals, bins=bins, weights=oc_weights,             label='Real OC',                   color='tab:red',    **kw)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Events')
            ax.set_yscale('log')
            ax.legend(fontsize=7)

    fig.tight_layout()
    plt.show()


from tqdm import tqdm
batch = next(iter(tqdm(dm.val_dataloader())))
plotting(batch)