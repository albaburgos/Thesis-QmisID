# Experimentation with AddCondKernel, which adds a conditioning kernel to a base kernel. 

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

path = "truth_dm"

if __name__ == "__main__":
    torch.manual_seed(1234)

    dm = PandasDirDataModule(
        dataset_dir=path,
        feature_cols=["truth_el_pt", "truth_el_eta"],
        target_cols=["pt_err"],
        #weight_col="weight",
        split=0.8,
        dataloader_kwargs={"num_workers": 8, "batch_size": 2**15},
    )
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    dm2 = PandasDirDataModule(
        dataset_dir=path,
        feature_cols=["truth_el_pt", "truth_el_eta"],
        target_cols=["eta_err"],
        #weight_col="weight",
        split=0.8,
        dataloader_kwargs={"num_workers": 8, "batch_size": 2**15},
    )
    train_loader2 = dm.train_dataloader()
    val_loader2 = dm.val_dataloader()

    eta_pt_encoding = TrivialEncoding(2)
    pt_base = GaussianKernel(eta_pt_encoding)
    eta_base = GaussianKernel(eta_pt_encoding)
    base = pt_base.__and__(eta_base)
    kernel = AddCondKernel(base)
    dielectron_kernel = kernel.__and__(kernel)
    
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=200,
        log_every_n_steps=50,
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min"),
            # EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
        logger=TensorBoardLogger(save_dir="shift_logs", name="kappa"),
        num_sanity_val_steps=0,
    )

    trainer.fit(pt_base, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.fit(eta_base, train_dataloaders=train_loader2, val_dataloaders=val_loader2)


    dielectron_kernel._draw(torch.Tensor([[0, 0]] * 10))
    