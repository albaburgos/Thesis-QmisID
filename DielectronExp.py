# Toy model for Kernel training with single-electron dataset

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
from iwpc.learn_dist.kernels.mixture_kernel import MixtureKernel
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.learn_dist.kernels.unlabelled_kernel_trainer import UnLabelledKernelTrainer
from iwpc.learn_dist.kernels.prepend_cond_kernel import PrependCondKernel
from iwpc.models.utils import basic_model_factory
from iwpc.models.layers import ConstantScaleLayer

if __name__ == "__main__":
    torch.manual_seed(1234)

    # One row per electron. SC (isQMisID=1) electrons are base samples;
    # OC (isQMisID=0) electrons are the real data distribution.
    dm = PandasDirDataModule(
        dataset_dir="/Users/albaburgosmondejar/Desktop/SingleElectrondelta",
        feature_spec=[
            ["eta", "pt"],                         # base_samples: SC [eta, truth_pt] → kernel conditioning
            ["eta", "pt", "pt_error", "isQMisID"], # data_samples: OC [eta, pt, pt_error, isQMisID=0]
            "isQMisID",                            # label: 0=OC (p), 1=SC (base q)
        ],
        weight_col="weight",
        split=0.8,
        dataloader_kwargs={"num_workers": 8, "batch_size": 2**15},
    )

    # Single-electron conditioning: eta and pt (truth_pt for base samples)
    kinematics_encoding = TrivialEncoding(2)
    shift_scale = np.log(3000)
    shift_loc = 0.0

    # Unflipped: Gaussian detector resolution smearing
    unflip_loc_model   = basic_model_factory(kinematics_encoding, 1,                      final_layers=[ConstantScaleLayer(shift_loc)])
    unflip_scale_model = basic_model_factory(kinematics_encoding, ExponentialEncoding(1), final_layers=[ConstantScaleLayer(shift_scale)])

    # Flipped: mixture of brem and resolution-flip Gaussians
    brem_loc_model       = basic_model_factory(kinematics_encoding, 1,                      final_layers=[ConstantScaleLayer(shift_loc)])
    brem_scale_model     = basic_model_factory(kinematics_encoding, ExponentialEncoding(1), final_layers=[ConstantScaleLayer(shift_scale)])
    flip_res_loc_model   = basic_model_factory(kinematics_encoding, 1,                      final_layers=[ConstantScaleLayer(shift_loc)])
    flip_res_scale_model = basic_model_factory(kinematics_encoding, ExponentialEncoding(1), final_layers=[ConstantScaleLayer(shift_scale)])

    unflip_smearing_kernel = GaussianKernel(kinematics_encoding, loc_model=unflip_loc_model,   scale_model=unflip_scale_model,   max_chi=5)
    brem_pt_err_kernel     = GaussianKernel(kinematics_encoding, loc_model=brem_loc_model,     scale_model=brem_scale_model,     max_chi=5)
    flip_res_pt_err_kernel = GaussianKernel(kinematics_encoding, loc_model=flip_res_loc_model, scale_model=flip_res_scale_model, max_chi=5)

    # Flipped electrons are a mixture of brem and resolution flip types
    flip_smearing_kernel = MixtureKernel([brem_pt_err_kernel, flip_res_pt_err_kernel], kinematics_encoding)

    # Branch on isQMisID: 0 → Gaussian resolution smearing, 1 → mixture (brem or res flip)
    # smearing_kernel: cond=[isQMisID, eta, pt] (3D), sample=pt_error (1D)
    smearing_kernel = BranchingKernel(
        [unflip_smearing_kernel, flip_smearing_kernel],
        [0,],
        lambda x: (x[:, 0] > 0).long()
    )

    # Discrete charge-flip decision: cond=[eta, pt] (2D), sample=isQMisID (1D)
    charge_flip_kernel = FiniteKernel(2, kinematics_encoding)

    # single_electron_kernel: cond=[eta, pt] (2D), sample=[pt_error, isQMisID] (2D)
    single_electron_kernel = smearing_kernel | charge_flip_kernel

    # Prepend conditioning so kernel output is [eta, pt, pt_error, isQMisID] (4D),
    # allowing the discriminator to condition on kinematics as well as pt_error.
    conditioned_kernel = PrependCondKernel(single_electron_kernel)

    # Discriminator: TrivialEncoding(3) passes [eta, pt, pt_error]; NopeEncoding(1) drops isQMisID.
    model = basic_model_factory(TrivialEncoding(3) & NopeEncoding(1))

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=200,
        log_every_n_steps=50,
        logger=TensorBoardLogger(save_dir="grad_logs", name="single_electron"),
        num_sanity_val_steps=0,
    )

    unlabelled = UnLabelledKernelTrainer(conditioned_kernel, model, 0.05, 8, 0.5, 5, 0.001, 0.0001)

    trainer.fit(unlabelled, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
