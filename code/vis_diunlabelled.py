# Script for IWPC visualisation of di-electron QmisID with kernel training

from pathlib import Path
import numpy as np
import torch
from torch import nn
from bokeh.io import curdoc

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
from _InvariantMass import CustomDielectronEncoding
from iwpc.learn_dist.kernels.partially_exact_unlabelled_kernel_trainer import PartiallyExactUnLabelledKernelTrainer
from iwpc.scalars.scalar import Scalar
from iwpc.scalars.scalar_function import ScalarFunction
from iwpc.visualise.bokeh_function_visualiser_2D import BokehFunctionVisualiser2D

DEVICE = torch.device("cpu")

# ── rebuild kernels (must match DielectronUnlabelled.py exactly) ─────────────
single_electron_encoding = TrivialEncoding(3)
shift_scale = np.log(1e3)
shift_loc = 0
# flip_init_log_probs = [np.log(0.98), np.log(0.01), np.log(0.01)]
single_electron_flip_kernel = FiniteKernel(3, single_electron_encoding)
event_flip_kernel = single_electron_flip_kernel + single_electron_flip_kernel

single_electron_no_flip_kernel = ConstantKernel([0], single_electron_encoding)
single_electron_brehm_kernel = GaussianKernel(
    single_electron_encoding,
    loc_model=basic_model_factory(single_electron_encoding, 1, final_layers=[ConstantScaleLayer(shift_loc)]),
    scale_model=basic_model_factory(single_electron_encoding, ExponentialEncoding(1), final_layers=[ConstantScaleLayer(shift_scale)]),
)
single_electron_res_kernel = GaussianKernel(
    single_electron_encoding,
    loc_model=basic_model_factory(single_electron_encoding, 1, final_layers=[ConstantScaleLayer(shift_loc)]),
    scale_model=basic_model_factory(single_electron_encoding, ExponentialEncoding(1), final_layers=[ConstantScaleLayer(shift_scale)]),
)
single_electron_eta_kernel   = ConstantKernel([0], TrivialEncoding(4))
single_electron_phi_kernel   = ConstantKernel([0], TrivialEncoding(4))
single_electron_pt_kernel = BranchingKernel.condition_on([
    single_electron_no_flip_kernel,
    single_electron_res_kernel,
    single_electron_brehm_kernel,
], single_electron_flip_kernel) & single_electron_eta_kernel & single_electron_phi_kernel

dielectron_smear_kernel = single_electron_pt_kernel + single_electron_pt_kernel

reordered_dielectron_smear_kernel = PermutationKernel(
    dielectron_smear_kernel,
    cond_permutation=[0, 2, 3, 4, 1, 5, 6, 7],
)

model = basic_model_factory(CustomDielectronEncoding(TrivialEncoding(2) & NopeEncoding(1) & TrivialEncoding(2) & NopeEncoding(3)))
unlabelled = PartiallyExactUnLabelledKernelTrainer(
    event_flip_kernel, reordered_dielectron_smear_kernel, model,
)

# ── load latest checkpoint ───────────────────────────────────────────────────
ckpt_candidates = sorted(
    Path("grad_logs/di_electron").glob("version_*/checkpoints/*.ckpt"),
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

# ── visualiser ───────────────────────────────────────────────────────────────
class DielectronKernelVisualiser(nn.Module):
    def __init__(self, trainer: PartiallyExactUnLabelledKernelTrainer, device: torch.device):
        super().__init__()
        self.trainer = trainer
        self.device = device

        # Per-lepton finite (flip) kernels.
        # event_flip_kernel is ConcatenatedKernel([l1_flip, l2_flip], concatenate_cond=True)
        self.l1_flip = trainer.exact_kernel.sub_kernels[0]
        self.l2_flip = trainer.exact_kernel.sub_kernels[1]

        # Per-lepton smearing: reordered → dielectron → [l1_pt_kernel, l2_pt_kernel]
        # each pt_kernel is ConcatenatedKernel([BranchingKernel, eta_kernel, phi_kernel], concatenate_cond=False)
        smear = trainer.sampled_kernel.base_kernel  # dielectron_smear_kernel
        l1_branch = smear.sub_kernels[0].sub_kernels[0]  # BranchingKernel for l1
        l2_branch = smear.sub_kernels[1].sub_kernels[0]  # BranchingKernel for l2

        # sub_kernels order matches condition_on([no_flip(0), res(1), brehm(2)], ...)
        self.l1_res_kernel   = l1_branch.sub_kernels[1]
        self.l1_brehm_kernel = l1_branch.sub_kernels[2]
        self.l2_res_kernel   = l2_branch.sub_kernels[1]
        self.l2_brehm_kernel = l2_branch.sub_kernels[2]

    def get_input_scalars(self):
        return [
            Scalar("l1_pt [MeV]", bins=np.linspace(0, 200_000, 200)),
            Scalar("l1_eta",      bins=np.linspace(-2.5, 2.5, 200)),
            Scalar("l1_phi",      bins=np.linspace(-np.pi, np.pi, 200)),
            Scalar("l2_pt [MeV]", bins=np.linspace(0, 200_000, 200)),
            Scalar("l2_eta",      bins=np.linspace(-2.5, 2.5, 200)),
            Scalar("l2_phi",      bins=np.linspace(-np.pi, np.pi, 200)),
        ]

    def get_output_scalars(self):
        return [
            ScalarFunction(lambda x: x["P_OC"],           "P(OC)"),
            ScalarFunction(lambda x: x["l1_P_no_flip"],   "l1 P(b=0, no flip)"),
            ScalarFunction(lambda x: x["l1_P_flip1"],     "l1 P(b=1, res)"),
            ScalarFunction(lambda x: x["l1_P_flip2"],     "l1 P(b=2, brehm)"),
            ScalarFunction(lambda x: x["l2_P_no_flip"],   "l2 P(b=0, no flip)"),
            ScalarFunction(lambda x: x["l2_P_flip1"],     "l2 P(b=1, res)"),
            ScalarFunction(lambda x: x["l2_P_flip2"],     "l2 P(b=2, brehm)"),
            ScalarFunction(lambda x: x["l1_res_loc"],     "l1 res loc"),
            ScalarFunction(lambda x: x["l1_res_sigma"],   "l1 res sigma"),
            ScalarFunction(lambda x: x["l1_brehm_loc"],   "l1 brehm loc"),
            ScalarFunction(lambda x: x["l1_brehm_sigma"], "l1 brehm sigma"),
            ScalarFunction(lambda x: x["l2_res_loc"],     "l2 res loc"),
            ScalarFunction(lambda x: x["l2_res_sigma"],   "l2 res sigma"),
            ScalarFunction(lambda x: x["l2_brehm_loc"],   "l2 brehm loc"),
            ScalarFunction(lambda x: x["l2_brehm_sigma"], "l2 brehm sigma"),
        ]

    def evaluate_for_visualiser(self, z: np.ndarray) -> dict:
        # z columns: [l1_pt, l1_eta, l1_phi, l2_pt, l2_eta, l2_phi]
        x = torch.tensor(z, dtype=torch.float32, device=self.device)
        x1 = x[:, :3]  # l1 kinematics: [pt, eta, phi]
        x2 = x[:, 3:]  # l2 kinematics: [pt, eta, phi]

        with torch.no_grad():
            # Flip probabilities from the per-lepton FiniteKernels
            l1_probs = self.l1_flip.construct_log_probs(x1).softmax(dim=-1).cpu().numpy()
            l2_probs = self.l2_flip.construct_log_probs(x2).softmax(dim=-1).cpu().numpy()

            # Smearing Gaussian parameters — sub-kernels see [q/pt, eta, phi] (3D)
            l1_res_loc     = self.l1_res_kernel.loc_model(x1).squeeze(-1).cpu().numpy()
            l1_res_sigma   = self.l1_res_kernel.scale_model(x1).squeeze(-1).cpu().numpy()
            l1_brehm_loc   = self.l1_brehm_kernel.loc_model(x1).squeeze(-1).cpu().numpy()
            l1_brehm_sigma = self.l1_brehm_kernel.scale_model(x1).squeeze(-1).cpu().numpy()

            l2_res_loc     = self.l2_res_kernel.loc_model(x2).squeeze(-1).cpu().numpy()
            l2_res_sigma   = self.l2_res_kernel.scale_model(x2).squeeze(-1).cpu().numpy()
            l2_brehm_loc   = self.l2_brehm_kernel.loc_model(x2).squeeze(-1).cpu().numpy()
            l2_brehm_sigma = self.l2_brehm_kernel.scale_model(x2).squeeze(-1).cpu().numpy()

            # Discriminator: q_obs = [l1_pt, l1_eta, l1_phi, l2_pt, l2_eta, l2_phi, b1, b2] (8D)
            # CustomDielectronEncoding wraps TrivialEncoding(2)&NopeEncoding(1)&TrivialEncoding(2)&NopeEncoding(3):
            # sees [l1_pt, l1_eta, l2_pt, l2_eta, invmass]; phi and b values are dropped.
            disc_input = torch.cat([x, torch.zeros(len(x), 2, device=self.device)], dim=1)
            log_p_over_q = self.trainer.log_p_over_q_model(disc_input)[:, 0]
            P_OC = torch.sigmoid(-log_p_over_q).cpu().numpy()

        return {
            "P_OC":           P_OC,
            "l1_P_no_flip":   l1_probs[:, 0],
            "l1_P_flip1":     l1_probs[:, 1],
            "l1_P_flip2":     l1_probs[:, 2],
            "l2_P_no_flip":   l2_probs[:, 0],
            "l2_P_flip1":     l2_probs[:, 1],
            "l2_P_flip2":     l2_probs[:, 2],
            "l1_res_loc":     l1_res_loc,
            "l1_res_sigma":   l1_res_sigma,
            "l1_brehm_loc":   l1_brehm_loc,
            "l1_brehm_sigma": l1_brehm_sigma,
            "l2_res_loc":     l2_res_loc,
            "l2_res_sigma":   l2_res_sigma,
            "l2_brehm_loc":   l2_brehm_loc,
            "l2_brehm_sigma": l2_brehm_sigma,
        }

    @property
    def center_point(self):
        # [l1_pt, l1_eta, l1_phi, l2_pt, l2_eta, l2_phi] at typical Z→ee kinematics
        return [1e5, 0.0, 0.0, 1e5, 0.0, 0.0]


module = DielectronKernelVisualiser(trainer=unlabelled, device=DEVICE)

curdoc().add_root(
    BokehFunctionVisualiser2D.visualise(
        module,
        initial_x_axis_scalar_ind=0,
        initial_y_axis_scalar_ind=1,
        initial_output_scalar_ind=0,
        selected_input_parameter_resolution=400,
    ).root
)
