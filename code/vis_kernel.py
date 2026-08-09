# Script for IWPC visualisation of high-dimensional Dielecton Kernel Training

from typing import Optional, Tuple, List
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn, Tensor
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from bokeh.io import curdoc

PROJECT_ROOT = Path(__file__).resolve().parent
IWPC_SRC = PROJECT_ROOT / "divergences" / "src"
if str(IWPC_SRC) not in sys.path:
    sys.path.insert(0, str(IWPC_SRC))

from iwpc.data_modules.pandas_data_module import BinaryPandasDataModule
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule

from iwpc.encodings.encoding_base import Encoding
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.encodings.continuous_periodic_encoding import ContinuousPeriodicEncoding
from iwpc.encodings.log_softmax_encoding import LogSoftmaxEncoding

from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.kernels.discrete_kernel import DiscreteKernel
from iwpc.learn_dist.kernels.constant_kernel import ConstantKernel
from iwpc.learn_dist.kernels.mixture_kernel import MixtureKernel
from iwpc.learn_dist.kernels.add_cond_kernel import AddCondKernel
from iwpc.learn_dist.kernels.trainable_kernel_base import (
    TrainableKernelBase,
    ConcatenatedKernel,
)
from iwpc.learn_dist.kernels.unlabelled_kernel_trainer import (
    UnLabelledKernelTrainer,
)

from iwpc.models.utils import basic_model_factory
from iwpc.models.layers import ConstantScaleLayer

from iwpc.scalars.scalar import Scalar
from iwpc.scalars.scalar_function import ScalarFunction
from iwpc.visualise.bokeh_function_visualiser_2D import (
    BokehFunctionVisualiser2D,
)
from discrete_branching import DiscreteBranchingKernel
from gaussian_kernel_single import make_kernel
from gaussian_kernel_variable_alpha import make_mixture_kernel
from divergences.src.iwpc.utils import latest_ckpt

DEVICE = torch.device("cpu")

CKPT_PATH = "kernel_SC/gaussian_mixture/version_9/checkpoints/epoch=79-step=1520.ckpt"

kernel = make_mixture_kernel(
    scale1=200,
    loc1=400,
    scale2=800,
    loc2=0,
    alpha_init=0.5,
    trainable_alpha=True,
)
ckpt_dict = torch.load(CKPT_PATH, map_location=DEVICE)
load_info = kernel.load_state_dict(ckpt_dict["state_dict"], strict=False)
kernel.cpu()

class KernelVisualiser2D(nn.Module):
    def __init__(self, kernel: nn.Module, device: torch.device):
        super().__init__()
        self.kernel = kernel
        self.device = device

    def get_input_scalars(self):
        return [
            Scalar("q_pt",  bins=np.linspace(0, 5e-4, 200)),
            Scalar("eta", bins=np.linspace(0.0, 2.5, 200)),
        ]

    def get_output_scalars(self):
        return [
            ScalarFunction(lambda x: x['mean1'], "mean1"),
            ScalarFunction(lambda x: x['sigma1'], "sigma1"),
            ScalarFunction(lambda x: x['mean2'], "mean2"),
            ScalarFunction(lambda x: x['sigma2'], "sigma2"),
            ScalarFunction(lambda x: x['alpha'], "alpha"),
        ]

    def evaluate_for_visualiser(self, z: np.ndarray):
        x = torch.tensor(z[:, :2], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            k1 = self.kernel.sub_kernels[0]
            k2 = self.kernel.sub_kernels[1]

            mean1 = k1.loc_model(x).squeeze(-1)
            sigma1 = k1.scale_model(x).squeeze(-1)
            mean2 = k2.loc_model(x).squeeze(-1)
            sigma2 = k2.scale_model(x).squeeze(-1)

            if hasattr(self.kernel, "alpha_model") and callable(self.kernel.alpha_model):
                alpha = torch.full_like(mean1, float(self.kernel.alpha_model()))
            else:
                log_probs = self.kernel.log_probability_model(x)
                alpha = torch.exp(log_probs[..., 0]).squeeze(-1)

        return {
            'mean1': mean1.cpu().numpy(),
            'sigma1': sigma1.cpu().numpy(),
            'mean2': mean2.cpu().numpy(),
            'sigma2': sigma2.cpu().numpy(),
            'alpha': alpha.cpu().numpy(), 
        }
    
    @property
    def center_point(self):
        return [45e3, 0.01]

module = KernelVisualiser2D(kernel=kernel, device=DEVICE)

curdoc().add_root(
    BokehFunctionVisualiser2D.visualise(
        module,
        initial_x_axis_scalar_ind=0,
        initial_y_axis_scalar_ind=0,
        initial_output_scalar_ind=0,
        selected_input_parameter_resolution=400,
    ).root
)
