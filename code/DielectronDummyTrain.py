# Experimentation: Attempt at training dielectron kernel model

from pathlib import Path
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.encodings.nope_encoding import NopeEncoding
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.kernels.branching_kernel import BranchingKernel
from iwpc.learn_dist.kernels.finite_kernel import FiniteKernel
from iwpc.learn_dist.kernels.permutation_kernel import PermutationKernel
from Xplot_callback import SmearedPtPlotCallback

from typing import Optional, Tuple, List
import torch
from torch import nn, Tensor
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.encodings.encoding_base import Encoding
from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.learn_dist.kernels.unlabelled_kernel_trainer import UnLabelledKernelTrainer
from iwpc.models.utils import basic_model_factory
from iwpc.models.layers import ConstantScaleLayer

if __name__ == "__main__":
    torch.manual_seed(1234)
    path = "/Users/albaburgosmondejar/Desktop/BenchmarkdeltaMC"

    dm = PandasDirDataModule(
        dataset_dir=path,
        feature_spec=[
            ["l1_eta", "l2_eta"],                                           # base_samples: kernel cond (2D)
            ["l1_pt_error", "l2_pt_error", "l1_isQMisID", "l2_isQMisID"],   # data_samples: kernel output space (4D)
            "label"
        ],
        weight_col="weight",
        split=0.8,
        dataloader_kwargs={"num_workers": 8, "batch_size": 2**15},
    )

    # Single-electron conditioning: eta only
    kinematics_encoding = TrivialEncoding(1)
    shift_scale = np.log(3000)
    shift_loc = 0.0

    # Separate models per Gaussian so resolution and charge-flip kernels learn independently
    res_loc_model    = basic_model_factory(kinematics_encoding, 1,                     final_layers=[ConstantScaleLayer(shift_loc)])
    res_scale_model  = basic_model_factory(kinematics_encoding, ExponentialEncoding(1), final_layers=[ConstantScaleLayer(shift_scale)])
    flip_loc_model   = basic_model_factory(kinematics_encoding, 1,                     final_layers=[ConstantScaleLayer(shift_loc)])
    flip_scale_model = basic_model_factory(kinematics_encoding, ExponentialEncoding(1), final_layers=[ConstantScaleLayer(shift_scale)])

    # Gaussian pt_error kernels: cond=eta (dim=1), sample=pt_error (dim=1)
    res_pt_err_kernel     = GaussianKernel(kinematics_encoding, loc_model=res_loc_model,  scale_model=res_scale_model,  max_chi=5)
    flipped_pt_err_kernel = GaussianKernel(kinematics_encoding, loc_model=flip_loc_model, scale_model=flip_scale_model, max_chi=5)

    # Branch on isQMisID (first cond col): 0 → resolution smearing, 1 → charge-flip smearing
    # smearing_kernel: cond=[isQMisID, eta] (dim=2), sample=pt_error (dim=1)
    smearing_kernel = BranchingKernel(
        [res_pt_err_kernel, flipped_pt_err_kernel],
        [0,],
        lambda x: (x[:, 0] > 0).long()
    )

    # Discrete charge-flip decision: cond=eta (dim=1), sample=isQMisID (dim=1)
    charge_flip_kernel = FiniteKernel(2, kinematics_encoding)

    # single_electron_kernel: cond=eta (dim=1), sample=[pt_error, isQMisID] (dim=2)
    single_electron_kernel = smearing_kernel | charge_flip_kernel

    # dielectron_kernel: cond=[l1_eta, l2_eta] (dim=2), sample=[l1_pt_error, l1_isQMisID, l2_pt_error, l2_isQMisID] (dim=4)
    dielectron_kernel = single_electron_kernel + single_electron_kernel
    # Reorder to [l1_pt_error, l2_pt_error, l1_isQMisID, l2_isQMisID] so continuous dims come first
    reordered_kernel = PermutationKernel(dielectron_kernel, [0, 2, 1, 3])

    dielectron_flip_kernel = charge_flip_kernel + charge_flip_kernel
    cut_dielectron_flip_kernel = dielectron_flip_kernel.cut(lambda x: x.sum() == 1)

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    dummy_dielectron = next(iter(val_loader))[0][:10000]

    callback = SmearedPtPlotCallback(
        dummy_dielectron=dummy_dielectron,
        cut_dielectron_flip_kernel=cut_dielectron_flip_kernel,
        out_dir=Path("/Users/albaburgosmondejar/QMISID/smearing_plots/dielectron_training"),
        every_n_epochs=1,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=200,
        log_every_n_steps=50,
        logger=TensorBoardLogger(save_dir="grad_logs", name="kappa"),
        callbacks=[callback],
        num_sanity_val_steps=0,
    )

    # Discriminator input: [l1_pt_error, l1_isQMisID, l2_pt_error, l2_isQMisID] (4D).
    # NopeEncoding masks the discrete isQMisID dims so the discriminator is forced to
    # distinguish p from q on pt_error alone — required for mathematical correctness.
    model = basic_model_factory(
        TrivialEncoding(2) & NopeEncoding(2)
    )

    unlabelled = UnLabelledKernelTrainer(reordered_kernel, model, 0.75, 8, 0.5, 5, 0.001, 0.00001)

    trainer.fit(unlabelled, train_dataloaders=train_loader, val_dataloaders=val_loader)
