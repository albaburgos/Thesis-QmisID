import lightning as L
import numpy as np
import torch
from torch import Tensor, nn

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.kernels.mixture_kernel import MixtureKernel
from iwpc.models.layers import ConstantScaleLayer
from iwpc.models.utils import basic_model_factory


def make_kernel(scale, loc, max_chi=5):
    eta_pt_encoding = TrivialEncoding(2)

    loc_model = basic_model_factory(
        eta_pt_encoding,
        1,
        initial_layers=[ConstantScaleLayer(loc)],
    )
    scale_model = basic_model_factory(
        eta_pt_encoding, 
        ExponentialEncoding(1), 
        initial_layers=[ConstantScaleLayer(np.log(scale))],)

    kernel = GaussianKernel(
        cond=eta_pt_encoding, 
        loc_model=loc_model,
        scale_model=scale_model,
        max_chi=max_chi
    )
    return kernel


def gaussian_wrapper(dataset_dir, scale, loc):

    dm = PandasDirDataModule(
        dataset_dir=dataset_dir,
        feature_spec=[["pt", "eta"], ["pt_err"]],
        #weight_col="weight",
        split=0.8,
        dataloader_kwargs={"num_workers": 8, "batch_size": 2**15},
    )
    
    train_loaderOC = dm.train_dataloader()
    val_loaderOC = dm.val_dataloader()

    pt_base = make_kernel(scale, loc)

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=100,
        log_every_n_steps=200,
        callbacks=[ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")],
        logger=TensorBoardLogger(save_dir="kernel_OC", name="gaussian"),
        num_sanity_val_steps=0,
    )

    trainer.fit(pt_base, train_dataloaders=train_loaderOC, val_dataloaders=val_loaderOC)


if __name__ == "__main__":
    torch.manual_seed(1234)

    gaussian_wrapper("PtErrOC", scale=200, loc=400)
