# 1D pT dielectron training, toy model

from iwpc.symmetries.symmetrized_model import SymmetrizedModel
from iwpc.symmetries.finite_group_action import FiniteGroupAction
from iwpc.symmetries.lambda_action import LambdaAction
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
from iwpc.learn_dist.kernels.constant_kernel import ConstantKernel
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.learn_dist.kernels.unlabelled_kernel_trainer import UnLabelledKernelTrainer
from iwpc.models.utils import basic_model_factory
from iwpc.models.layers import ConstantScaleLayer
from iwpc.learn_dist.fdivergence_minimization.fdivergence_minimizing_kernel_trainer import FDivergenceMinimizingKernelTrainer
from iwpc.learn_dist.kernels.permutation_kernel import PermutationKernel
from _InvariantMass import CustomDielectronEncoding
from iwpc.divergences import DifferentiableFDivergence, KLDivergence, JensenShannonDivergence
from iwpc.learn_dist.kernels.finite_kernel import FiniteKernelInterface
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
import matplotlib.pyplot as plt


class DielectronKernelTrainer(FDivergenceMinimizingKernelTrainer):
    def __init__(self, *args, plot_every_n_steps: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_every_n_steps = plot_every_n_steps
        self._sc_full:         np.ndarray | None = None
        self._sc_weights_full: np.ndarray | None = None

    def setup(self, stage: str) -> None:
        if stage != 'fit':
            return
        buf, wbuf = [], []
        for batch in self.trainer.datamodule.val_dataloader():
            _, samples, labels, weights = batch
            p_mask = labels == 0
            if p_mask.any():
                buf.append(samples[p_mask, :2].numpy())
                wbuf.append(weights[p_mask].numpy())
        if buf:
            self._sc_full         = np.concatenate(buf)
            self._sc_weights_full = np.concatenate(wbuf)

    def _log_flip_plot(self, batch, global_step: int) -> None:
        if self.logger is None or self._sc_full is None:
            return

        base_samples, _, labels, weights = batch
        q_mask    = labels == 1
        q_base    = base_samples[q_mask]
        q_weights = weights[q_mask]

        with torch.no_grad():
            flip_prob = self.exact_kernel.cut_pass_log_prob(q_base).exp()

        oc      = q_base.cpu().numpy()          
        sc_full = self._sc_full                 
        kernel_weights = (q_weights * flip_prob).cpu().numpy()
        log_p_over_q_weights = (self.log_p_over_q_model(torch.cat([q_base, torch.zeros(q_base.shape[0], 2, device=q_base.device)], dim=-1)).detach().squeeze(-1).exp().cpu().numpy()) * q_weights.cpu().numpy()
        def inv_mass(x):
            pt1, eta1, phi1, pt2, eta2, phi2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
            return np.sqrt(np.maximum(2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2)), 0))

        kw = dict(histtype='step', linewidth=1.5, density=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for ax, oc_vals, sc_vals, bins, xlabel in [
            (axes[0], oc[:, 0], sc_full[:, 0], np.linspace(0,    1.5e5, 60), 'l1_pt [MeV]'),
            (axes[1], oc[:, 1], sc_full[:, 1], np.linspace(-2.5, 2.5,   60), 'l1_eta'),
            # (axes[2], oc[:, 3], sc_full[:, 3], np.linspace(0,    1.5e5, 60), 'l2_pt [MeV]'),
            # (axes[3], oc[:, 4], sc_full[:, 4], np.linspace(-2.5, 2.5,   60), 'l2_eta'),
            # (axes[4], inv_mass(oc), inv_mass(sc_full), np.linspace(6e4, 1.2e5, 60), 'm_l1l2 [MeV]'),
            # ( axes, oc[:, 0], sc_full[:, 0],  np.linspace(0,    1.5e5, 60), 'l1_pt [MeV]'),
        ]:
            ax.hist(oc_vals, bins=bins, weights=kernel_weights,       label='Kernel (OC × P(flip))', color='tab:green',  **kw)
            ax.hist(sc_vals, bins=bins, weights=self._sc_weights_full, label='Real SC',               color='tab:orange', **kw)
            ax.hist(oc_vals, bins=bins, weights=log_p_over_q_weights, label='Logp/q',               color='tab:red', **kw)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Density')
            ax.legend(fontsize=7)

        fig.tight_layout()
        self.logger.experiment.add_figure('flip_vs_kinematics', fig, global_step=global_step)
        plt.close(fig)

    def on_train_batch_end(self, *args) -> None:
        if self.global_step % self.plot_every_n_steps == 0:
            self._log_flip_plot(args[1], self.global_step)


if __name__ == "__main__":
    torch.manual_seed(1234)

    dm = PandasDirDataModule(
        dataset_dir="/Users/albaburgosmondejar/Desktop/BenchmarkMC",
        feature_spec=[
            # ["l1_pt", "l1_eta", "l1_phi", "l2_pt", "l2_eta", "l2_phi"],  # base: OC kinematics (6D), fed into exact_kernel
            # ["l1_pt", "l1_eta","l1_phi", "l2_pt", "l2_eta", "l2_phi", "b", "b"], # data: SC observables (8D), fed into discriminator
            ["l1_pt", "l1_eta"], 
            [ "l1_pt", "l1_eta", "b"], 
            'opposite_charge',
        ],
        weight_col="weight",
        split=0.8,
        dataloader_kwargs={"num_workers": 0, "batch_size": 2**15},
        # use_in_memory_dataset=True,
    )

    single_electron_encoding = TrivialEncoding(2)
    shift_scale = np.log(1e2)
    shift_loc = -1.2e3
   
    # single_electron_flip_kernel = FiniteKernel(3, single_electron_encoding, init_log_probs=[np.log(0.998), np.log(0.001), np.log(0.001)])
    # event_flip_kernel = single_electron_flip_kernel + single_electron_flip_kernel
    # exactly_one_flip_kernel = event_flip_kernel.cut(lambda x: (x.sum(dim=-1) > 0) and (x[0] == 0 or x[1] == 0))

    single_electron_no_flip_kernel = ConstantKernel([0], single_electron_encoding)
    single_electron_brehm_kernel = GaussianKernel(
        single_electron_encoding,
        loc_model=basic_model_factory(single_electron_encoding, 1, final_layers=[ConstantScaleLayer(shift_loc, np.exp(shift_scale))]),
        scale_model=basic_model_factory(single_electron_encoding, ExponentialEncoding(2), final_layers=[ConstantScaleLayer(shift_scale)]),
    )
    single_electron_res_kernel = GaussianKernel(
        single_electron_encoding,
        loc_model=basic_model_factory(single_electron_encoding, 1, final_layers=[ConstantScaleLayer(shift_loc, np.exp(shift_scale))]),
        scale_model=basic_model_factory(single_electron_encoding, ExponentialEncoding(2), final_layers=[ConstantScaleLayer(shift_scale)]),
    )

    # single_electron_eta_kernel = ConstantKernel([0], TrivialEncoding(4))
    # single_electron_phi_kernel = ConstantKernel([0], TrivialEncoding(4))
    # single_electron_pt_kernel = BranchingKernel.condition_on([
    #     single_electron_no_flip_kernel,
    #     single_electron_res_kernel,
    #     single_electron_brehm_kernel,
    # ], single_electron_flip_kernel) & single_electron_eta_kernel & single_electron_phi_kernel 

    # dielectron_smear_kernel = single_electron_pt_kernel + single_electron_pt_kernel

    # reordered_dielectron_smear_kernel = PermutationKernel(
    #     dielectron_smear_kernel,
    #     cond_permutation=[0, 2, 3, 4, 1,5, 6, 7],
    # )

    single_electron_flip_kernel = FiniteKernel(3, TrivialEncoding(2), init_log_probs=[np.log(0.998), np.log(0.001), np.log(0.001)])
    single_electron_one_flip_kernel = single_electron_flip_kernel.cut(lambda x: (x > 0))
    single_electron_eta_kernel = ConstantKernel([0], TrivialEncoding(3))
    model = basic_model_factory(TrivialEncoding(2)&NopeEncoding(1))

    single_electron_pt_kernel = BranchingKernel.condition_on([
        single_electron_no_flip_kernel,
        single_electron_res_kernel,
        single_electron_brehm_kernel,
    ], single_electron_flip_kernel) & single_electron_eta_kernel

    # print(single_electron_pt_kernel.cond_dimension, single_electron_one_flip_kernel.sample_dimension, single_electron_one_flip_kernel.cond_dimension)
    # full_kernel = single_electron_pt_kernel | single_electron_one_flip_kernel
    # print(full_kernel.sample_dimension)
    # print(full_kernel.draw(torch.zeros(10, 1)))

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=200,
        log_every_n_steps=1,
        logger=TensorBoardLogger(save_dir="div_logs", name="di_electron"),
        num_sanity_val_steps=0,
    )

    unlabelled = DielectronKernelTrainer(
    # unlabelled = DielectronKernelTrainer.load_from_checkpoint(
        # "div_logs/di_electron/version_53/checkpoints/epoch=0-step=535.ckpt",
        # strict=False,
        exact_kernel=single_electron_one_flip_kernel,
        sampled_kernel=single_electron_pt_kernel,
        log_p_over_q_model=model,
        start_kernel_train_epoch = 10,
        start_discriminator_train_epoch = 0,
        kernel_opt_lr=1e-4,
        divergence=KLDivergence()
    )

    trainer.fit(
        unlabelled,
        datamodule=dm,
    )