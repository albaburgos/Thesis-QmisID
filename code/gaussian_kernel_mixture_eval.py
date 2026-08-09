
from typing import Optional, Tuple
import torch
from torch import Tensor
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.kernels.trainable_kernel_base import ConcatenatedKernel
from iwpc.models.utils import basic_model_factory
from iwpc.learn_dist.kernels.add_cond_kernel import AddCondKernel
from typing import Optional, Tuple, List
import torch
from torch import Tensor
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from iwpc.data_modules.pandas_data_module import BinaryPandasDataModule
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.kernels.discrete_kernel import DiscreteKernel
from iwpc.learn_dist.kernels.trainable_kernel_base import (
    TrainableKernelBase,
    ConcatenatedKernel)
from torch.nn import Module
from iwpc.encodings.encoding_base import Encoding
from iwpc.models.utils import basic_model_factory
from iwpc.learn_dist.kernels.add_cond_kernel import AddCondKernel
from iwpc.learn_dist.kernels.constant_kernel import ConstantKernel
from iwpc.learn_dist.kernels.mixture_kernel import MixtureKernel
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.encodings.continuous_periodic_encoding import ContinuousPeriodicEncoding
from iwpc.encodings.log_softmax_encoding import LogSoftmaxEncoding
from iwpc.learn_dist.kernels.unlabelled_kernel_trainer import UnLabelledKernelTrainer
from TMTrain import KappaLightning
from discrete_branching import DiscreteBranchingKernel

if __name__ == "__main__":
    torch.manual_seed(1234)

    ckpt_path = "shift_logs/kappa/version_2/checkpoints/epoch=82-step=152056.ckpt"
    dm_classifier = KappaLightning.load_from_checkpoint(ckpt_path).eval()

    ckpt_path = "kernel_SC/gaussian/version_18/checkpoints/epoch=1-step=3664.ckpt"
    kernel_OC = GaussianKernel.load_from_checkpoint(ckpt_path, cond =TrivialEncoding(2))
    kernelOC = kernel_OC.eval()
    device = kernelOC.device
    print(kernelOC.draw(torch.tensor([[0,0]]*10,  device=device)))

    eta_pt_encoding = TrivialEncoding(2)
    branching_kernel = DiscreteBranchingKernel(
    eta_pt_encoding,
    DiscreteKernel(2, TrivialEncoding(1), dm_classifier),
    [kernelOC, kernelOC],
    explicit_mixture_label=True,
)
    logit_model: torch.nn.Module | None = None,
    print(branching_kernel.draw(torch.tensor([[0,0]]*2000000,  device=device)))
