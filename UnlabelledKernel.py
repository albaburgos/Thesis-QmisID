# Experimentation: Unlabelled Kernel Training for pt and eta distributions.


from typing import Optional, Tuple, List
import numpy as np
import torch
from torch import nn, Tensor
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from bokeh.io import curdoc
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
    ConcatenatedKernel)
from iwpc.learn_dist.kernels.unlabelled_kernel_trainer import (UnLabelledKernelTrainer)
from iwpc.models.utils import basic_model_factory
from iwpc.models.layers import ConstantScaleLayer
from iwpc.scalars.scalar import Scalar
from iwpc.scalars.scalar_function import ScalarFunction
from iwpc.visualise.bokeh_function_visualiser_2D import (
    BokehFunctionVisualiser2D)
from TMTrain import KappaLightning
from discrete_branching import DiscreteBranchingKernel
from gaussian_kernel_single import gaussian_wrapper, make_kernel
from divergences.src.iwpc.utils import latest_ckpt

path = "PtErrUnlabelled"

if __name__ == "__main__":
    torch.manual_seed(1234)
    
    dm = PandasDirDataModule(
        dataset_dir=path, 
        feature_spec=[["pt", "eta"], ["label"]],
        #weight_col="weight",
        split=0.8,
        dataloader_kwargs={"num_workers": 8, "batch_size": 2**15},
    )

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=200,
        log_every_n_steps=50,
        # #callbacks=[
        #     ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min"),
        # ],
        logger=TensorBoardLogger(save_dir="grad_logs", name="kappa"),
        num_sanity_val_steps=0,
    )

    eta_pt_encoding = TrivialEncoding(2)
    
    shift_scale = np.log(3000)
    shift_loc = 200
    
    loc_model = basic_model_factory(
        eta_pt_encoding,
        1,
        initial_layers=[ConstantScaleLayer(shift_loc)],
    )
    scale_model = basic_model_factory(
        eta_pt_encoding, 
        ExponentialEncoding(1), 
        initial_layers=[ConstantScaleLayer(shift_scale)],)

    pt_base = GaussianKernel(cond=eta_pt_encoding,loc_model = loc_model,scale_model=scale_model, max_chi=5)
    eta_base = GaussianKernel(cond = eta_pt_encoding, loc_model = loc_model, scale_model=scale_model, max_chi=5)
    base = pt_base & eta_base
     
    kernel = AddCondKernel(base)
   
    model = basic_model_factory( TrivialEncoding(2), TrivialEncoding(1))
        
    unlabelled = UnLabelledKernelTrainer(kernel, model)

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    trainer.fit(unlabelled, train_dataloaders=train_loader, val_dataloaders=val_loader)
    dielectron_kernel = kernel & kernel
