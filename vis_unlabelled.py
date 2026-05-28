# IWPC visualisation script for single-electron QmisID probability with kernel training

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
from iwpc.learn_dist.kernels.mixture_kernel import MixtureKernel
from iwpc.learn_dist.kernels.unlabelled_kernel_trainer import UnLabelledKernelTrainer
from iwpc.learn_dist.kernels.prepend_cond_kernel import PrependCondKernel
from iwpc.models.utils import basic_model_factory
from iwpc.models.layers import ConstantScaleLayer
from iwpc.scalars.scalar import Scalar
from iwpc.scalars.scalar_function import ScalarFunction
from iwpc.visualise.bokeh_function_visualiser_2D import BokehFunctionVisualiser2D

DEVICE = torch.device("cpu")

# ── rebuild kernel (must match SingleUnlabelled2.py exactly) ─────────────────
kinematics_encoding = TrivialEncoding(2)
shift_scale = np.log(3000)
shift_loc   = 0.0

unflip_loc_model   = basic_model_factory(kinematics_encoding, 1,                      final_layers=[ConstantScaleLayer(shift_loc)])
unflip_scale_model = basic_model_factory(kinematics_encoding, ExponentialEncoding(1), final_layers=[ConstantScaleLayer(shift_scale)])

brem_loc_model       = basic_model_factory(kinematics_encoding, 1,                      final_layers=[ConstantScaleLayer(shift_loc)])
brem_scale_model     = basic_model_factory(kinematics_encoding, ExponentialEncoding(1), final_layers=[ConstantScaleLayer(shift_scale)])
flip_res_loc_model   = basic_model_factory(kinematics_encoding, 1,                      final_layers=[ConstantScaleLayer(shift_loc)])
flip_res_scale_model = basic_model_factory(kinematics_encoding, ExponentialEncoding(1), final_layers=[ConstantScaleLayer(shift_scale)])

unflip_smearing_kernel = GaussianKernel(kinematics_encoding, loc_model=unflip_loc_model,   scale_model=unflip_scale_model,   max_chi=5)
brem_pt_err_kernel     = GaussianKernel(kinematics_encoding, loc_model=brem_loc_model,     scale_model=brem_scale_model,     max_chi=5)
flip_res_pt_err_kernel = GaussianKernel(kinematics_encoding, loc_model=flip_res_loc_model, scale_model=flip_res_scale_model, max_chi=5)

flip_smearing_kernel  = MixtureKernel([brem_pt_err_kernel, flip_res_pt_err_kernel], kinematics_encoding)

smearing_kernel = BranchingKernel(
    [unflip_smearing_kernel, flip_smearing_kernel],
    [0,],
    lambda x: (x[:, 0] > 0).long()
)

charge_flip_kernel     = FiniteKernel(2, kinematics_encoding)
single_electron_kernel = smearing_kernel | charge_flip_kernel
conditioned_kernel     = PrependCondKernel(single_electron_kernel)

model      = basic_model_factory(TrivialEncoding(3) & NopeEncoding(1))
unlabelled = UnLabelledKernelTrainer(conditioned_kernel, model, 0.05, 8, 0.5, 5, 0.001, 0.0001)

# ── load latest checkpoint ───────────────────────────────────────────────────
ckpt_candidates = sorted(
    Path("grad_logs/single_electron").glob("version_*/checkpoints/*.ckpt"),
    key=lambda p: p.stat().st_mtime,
)
if not ckpt_candidates:
    raise FileNotFoundError("No checkpoint found under grad_logs/single_electron/")
CKPT_PATH = ckpt_candidates[-1]
print(f"Loading checkpoint: {CKPT_PATH}")

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
missing, unexpected = unlabelled.load_state_dict(ckpt["state_dict"], strict=False)
if missing:
    print(f"WARNING: {len(missing)} keys not loaded (architecture mismatch — retrain to fix P_SC output): {missing[:3]}...")
unlabelled.eval()

# ── visualiser ───────────────────────────────────────────────────────────────
class SingleElectronKernelVisualiser(nn.Module):
    def __init__(self, trainer: UnLabelledKernelTrainer, device: torch.device):
        super().__init__()
        self.trainer = trainer
        self.device  = device

        # shortcuts into the kernel tree (trainer.kernel is PrependCondKernel wrapping ConditionedKernel)
        kernel = trainer.kernel.kernel                                 # ConditionedKernel
        self.charge_flip      = kernel.conditioning_kernel             # FiniteKernel
        self.unflip_kernel    = kernel.sample_kernel.sub_kernels[0]   # GaussianKernel (unflipped res)
        self.flip_smearing    = kernel.sample_kernel.sub_kernels[1]   # MixtureKernel
        self.brem_kernel      = self.flip_smearing.sub_kernels[0]     # GaussianKernel (brem)
        self.flip_res_kernel  = self.flip_smearing.sub_kernels[1]     # GaussianKernel (res flip)

    def get_input_scalars(self):
        return [
            Scalar("eta",            bins=np.linspace(-2.5, 2.5,      200)),
            Scalar("pt [MeV]",       bins=np.linspace(0,   300_000,   200)),
            Scalar("pt_error [MeV]", bins=np.linspace(-50_000, 50_000, 200)),
        ]

    def get_output_scalars(self):
        return [
            ScalarFunction(lambda x: x["unflip_loc"],     "unflip loc"),
            ScalarFunction(lambda x: x["unflip_sigma"],   "unflip sigma"),
            ScalarFunction(lambda x: x["brem_loc"],       "brem loc"),
            ScalarFunction(lambda x: x["brem_sigma"],     "brem sigma"),
            ScalarFunction(lambda x: x["flip_res_loc"],   "flip_res loc"),
            ScalarFunction(lambda x: x["flip_res_sigma"], "flip_res sigma"),
            ScalarFunction(lambda x: x["brem_fraction"],  "brem fraction"),
            ScalarFunction(lambda x: x["P_SC"],           "P(SC | eta, pt, pt_error)"),
        ]

    def evaluate_for_visualiser(self, z: np.ndarray) -> dict:
        x        = torch.tensor(z[:, :2], dtype=torch.float32, device=self.device)
        pt_error = torch.tensor(z[:, 2],  dtype=torch.float32, device=self.device)

        with torch.no_grad():
            unflip_loc   = self.unflip_kernel.loc_model(x).squeeze(-1).cpu().numpy()
            unflip_sigma = self.unflip_kernel.scale_model(x).squeeze(-1).cpu().numpy()

            brem_loc   = self.brem_kernel.loc_model(x).squeeze(-1).cpu().numpy()
            brem_sigma = self.brem_kernel.scale_model(x).squeeze(-1).cpu().numpy()

            flip_res_loc   = self.flip_res_kernel.loc_model(x).squeeze(-1).cpu().numpy()
            flip_res_sigma = self.flip_res_kernel.scale_model(x).squeeze(-1).cpu().numpy()

            # mixture weight: brem_fraction = P(brem | eta, pt)
            mix_log_probs = self.flip_smearing.log_probability_model(x)
            brem_fraction = mix_log_probs[:, 0].exp().cpu().numpy()

            # P(SC | eta, pt, pt_error): discriminator outputs log(p_OC / q_SC),
            # so sigmoid(-output) = q/(p+q) = P(SC).
            # Input: [eta, pt, pt_error, isQMisID=0]; NopeEncoding drops the last dim.
            disc_input   = torch.cat([x, pt_error.unsqueeze(1), torch.zeros(len(x), 1, device=self.device)], dim=1)
            log_p_over_q = self.trainer.log_p_over_q_model(disc_input)[:, 0]
            P_SC         = torch.sigmoid(-log_p_over_q).cpu().numpy()

        return {
            "unflip_loc":     unflip_loc,
            "unflip_sigma":   unflip_sigma,
            "brem_loc":       brem_loc,
            "brem_sigma":     brem_sigma,
            "flip_res_loc":   flip_res_loc,
            "flip_res_sigma": flip_res_sigma,
            "brem_fraction":  brem_fraction,
            "P_SC":           P_SC,
        }

    @property
    def center_point(self):
        return [0.0, 50_000, 0.0]


module = SingleElectronKernelVisualiser(trainer=unlabelled, device=DEVICE)

curdoc().add_root(
    BokehFunctionVisualiser2D.visualise(
        module,
        initial_x_axis_scalar_ind=2,
        initial_y_axis_scalar_ind=0,
        initial_output_scalar_ind=7,
        selected_input_parameter_resolution=400,
    ).root
)
