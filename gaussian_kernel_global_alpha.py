# Experimentation training a gaussian mixture kernel to learn pt_errors, with a trainable global alpha parameter

import lightning as L
import inspect
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.kernels.mixture_kernel import MixtureKernel
from iwpc.models.layers import ConstantScaleLayer
from iwpc.models.utils import basic_model_factory
from gaussian_kernel_single import make_kernel

# global alpha training 

class AlphaLogProbabilityModel(nn.Module):
    """
    Global 2-component mixture weights:
    p(K1)=alpha, p(K2)=1-alpha.
    """
    def __init__(self, alpha_init: float = 0.5, trainable_alpha: bool = True):
        super().__init__()

        init_logit = torch.logit(torch.tensor(alpha_init, dtype=torch.float32))
        if trainable_alpha:
            self.alpha_logit = nn.Parameter(init_logit)
        else:
            self.register_buffer("alpha_logit", init_logit)

    def forward(self, cond: Tensor) -> Tensor:
        log_probs = torch.stack(
            [F.logsigmoid(self.alpha_logit), F.logsigmoid(-self.alpha_logit)],
            dim=0,
        )
        return log_probs.unsqueeze(0).expand(cond.shape[0], -1)

    def alpha(self) -> float:
        return float(torch.sigmoid(self.alpha_logit).detach().cpu())


def make_mixture_kernel(
    scale1,
    loc1,
    scale2,
    loc2,
    alpha_init: float = 0.5,
    trainable_alpha: bool = True,
):
    k1 = make_kernel(scale1, loc1, max_chi=None)
    k2 = make_kernel(scale2, loc2, max_chi=None)
    eta_pt_encoding = TrivialEncoding(2)
    alpha_model = AlphaLogProbabilityModel(
        alpha_init=alpha_init,
        trainable_alpha=trainable_alpha,
    )
    mixture = MixtureKernel(
        cond=eta_pt_encoding,
        sub_kernels=[k1, k2],
        log_probability_model=alpha_model,
    )
    return mixture


def make_data_module(dataset_dir, split=0.8, batch_size=2**14, num_workers=8):
    sig = inspect.signature(PandasDirDataModule.__init__)
    kwargs = {
        "dataset_dir": dataset_dir,
        "split": split,
        "dataloader_kwargs": {"num_workers": num_workers, "batch_size": batch_size},
    }
    if "feature_spec" in sig.parameters:
        kwargs["feature_spec"] = [["pt", "eta"], ["pt_err"]]
    else:
        kwargs["feature_cols"] = ["pt", "eta"]
        kwargs["target_cols"] = ["pt_err"]

    return PandasDirDataModule(**kwargs)


def gaussian_mixture_wrapper(
    dataset_dir,
    scale1,
    loc1,
    scale2,
    loc2,
    alpha_init: float = 0.5,
    trainable_alpha: bool = True,
):
    dm = make_data_module(
        dataset_dir=dataset_dir,
        split=0.8,
        batch_size=2**15,
        num_workers=8,
    )

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    pt_base = make_mixture_kernel(
        scale1=scale1,
        loc1=loc1,
        scale2=scale2,
        loc2=loc2,
        alpha_init=alpha_init,
        trainable_alpha=trainable_alpha,
    )

    checkpoint_cb = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=100,
        log_every_n_steps=200,
        callbacks=[checkpoint_cb],
        logger=TensorBoardLogger(save_dir="kernel_SC", name="gaussian_mixture"),
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(pt_base, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return pt_base


if __name__ == "__main__":
    torch.manual_seed(1234)

    gaussian_mixture_wrapper(
        "PtErrSC",
        scale1=200,
        loc1=400,
        scale2=800,
        loc2=0,
        alpha_init=0.5,
        trainable_alpha=True,
    )
