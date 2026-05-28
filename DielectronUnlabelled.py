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
from iwpc.learn_dist.kernels.finite_cut_kernel import FiniteCutKernel
from iwpc.learn_dist.kernels.mixture_kernel import MixtureKernel
from iwpc.learn_dist.kernels.constant_kernel import ConstantKernel
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.learn_dist.kernels.unlabelled_kernel_trainer import UnLabelledKernelTrainer
from iwpc.models.utils import basic_model_factory
from iwpc.models.layers import ConstantScaleLayer
from iwpc.learn_dist.fdivergence_minimization.fdivergence_minimizing_kernel_trainer import FDivergenceMinimizingKernelTrainer
from iwpc.learn_dist.kernels.permutation_kernel import PermutationKernel
from InvariantMass import CustomDielectronEncoding
from iwpc.divergences import DifferentiableFDivergence, KLDivergence, JensenShannonDivergence
from iwpc.learn_dist.kernels.finite_kernel import FiniteKernelInterface
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
import matplotlib.pyplot as plt

class DielectronKernelTrainer(FDivergenceMinimizingKernelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _log_flip_plot(self, batch, global_step: int) -> None:

        base_samples, data_samples, labels, weights = batch
        q_mask    = labels == 1
        q_base    = base_samples[q_mask]
        sc = base_samples[~q_mask].cpu().numpy()
        sc_weights = weights[~q_mask].cpu().numpy()
        oc = base_samples[q_mask].cpu().numpy()
        oc_weights = weights[q_mask].cpu().numpy()

        simulated_all   = []
        event_probs_all = []
        model_ratio_all = []

        with torch.no_grad():
            exact_outcome_log_prob_iter, cut_pass_log_prob = self.exact_kernel.outcome_with_log_prob_iter_and_cut_pass_log_prob(q_base)
            flip_prob = cut_pass_log_prob.exp().cpu().numpy()
            for exact_outcome, exact_log_prob in exact_outcome_log_prob_iter:
                event_prob  = oc_weights * flip_prob * exact_log_prob.exp().cpu().numpy()
                repeated_exact_outcome = exact_outcome.repeat((q_base.shape[0], 1))
                sampled_kernel_cond = torch.concat([repeated_exact_outcome, q_base], dim=1)
                samples, _ = self.sampled_kernel.draw_with_log_prob(sampled_kernel_cond)
                simulated = q_base + samples
                model_ratio = oc_weights * self.log_p_over_q_model(
                    torch.concat([simulated, repeated_exact_outcome], dim=-1)
                ).exp().cpu().numpy()

                simulated_all.append(simulated.cpu().numpy())
                event_probs_all.append(event_prob)
                model_ratio_all.append(model_ratio)

        simulated_np  = np.concatenate(simulated_all)    # [2*N_oc, 6]
        event_probs   = np.concatenate(event_probs_all)  # [2*N_oc]
        model_weights = np.concatenate(model_ratio_all)  # [2*N_oc]
        oc_tiled      = np.tile(oc, (2, 1))              # [2*N_oc, 6]

        def inv_mass(x):
            pt1, eta1, phi1, pt2, eta2, phi2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
            return np.sqrt(np.maximum(2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2)), 0))

        kw = dict(histtype='step', linewidth=1.5)
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        for ax, oc_vals, oc_t, sc_vals, sim_vals, bins, xlabel in [
            (axes[0], oc[:, 0], oc_tiled[:, 0], sc[:, 0], simulated_np[:, 0], np.linspace(0,    1.5e5, 60), 'l1_pt [MeV]'),
            (axes[1], oc[:, 1], oc_tiled[:, 1], sc[:, 1], simulated_np[:, 1], np.linspace(-2.5, 2.5,   60), 'l1_eta'),
            (axes[2], oc[:, 3], oc_tiled[:, 3], sc[:, 3], simulated_np[:, 3], np.linspace(0,    1.5e5, 60), 'l2_pt [MeV]'),
            (axes[3], oc[:, 4], oc_tiled[:, 4], sc[:, 4], simulated_np[:, 4], np.linspace(-2.5, 2.5,   60), 'l2_eta'),
            (axes[4], inv_mass(oc), inv_mass(oc_tiled), inv_mass(sc), inv_mass(simulated_np), np.linspace(6e4, 1.2e5, 60), 'm_l1l2 [MeV]'),
        ]:
            ax.hist(oc_vals, bins=bins, weights=flip_prob * oc_weights, label='OC × P(flip) × w',        color='tab:green',  **kw)
            ax.hist(oc_t,    bins=bins, weights=event_probs,            label='OC × P(outcome) × w',     color='tab:blue',   **kw)
            ax.hist(sc_vals, bins=bins, weights=sc_weights,             label='Real SC',                  color='tab:orange', **kw)
            # ax.hist(sim_vals, bins=bins, weights=model_weights,         label='OC simulated × p/q × w',  color='tab:red',    **kw)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Events')
            ax.legend(fontsize=7)

        fig.tight_layout()
        self.logger.experiment.add_figure('flip_vs_kinematics', fig, global_step=global_step)
        plt.close(fig)

    def on_train_batch_end(self, *args) -> None:
        steps_per_epoch = self.trainer.num_training_batches
        half = max(1, steps_per_epoch // 10)
        if self.global_step % half == 0:
            self._log_flip_plot(args[1], self.global_step)

if __name__ == "__main__":
    torch.manual_seed(1234)

    dm = PandasDirDataModule(
        dataset_dir="/Users/albaburgosmondejar/Desktop/BenchmarkMC",
        feature_spec=[
            ["l1_pt", "l1_eta", "l1_phi", "l2_pt", "l2_eta", "l2_phi"],  # base: OC kinematics (6D), fed into exact_kernel
            ["l1_pt", "l1_eta","l1_phi", "l2_pt", "l2_eta", "l2_phi", "b", "b"], # data: SC observables (8D), fed into discriminator
            'opposite_charge',
        ],
        weight_col="weight",
        split=0.8,
        dataloader_kwargs={"num_workers": 0, "batch_size": 2**15},
        # use_in_memory_dataset=True,
    )

    single_electron_encoding = TrivialEncoding(3)
    shift_scale = np.log(1e3)
    shift_loc = -1.2e3
   
    single_electron_flip_kernel = FiniteKernel(2, single_electron_encoding, init_log_probs=[np.log(0.9998), np.log(0.0002)])
    event_flip_kernel = single_electron_flip_kernel + single_electron_flip_kernel
    # exactly_one_flip_kernel = event_flip_kernel.cut(lambda x: (x.sum(dim=-1) > 0) and (x[0] == 0 or x[1] == 0))
    exactly_one_flip_kernel = FiniteCutKernel(event_flip_kernel, lambda x: (x.sum(dim=-1) > 0) and (x[0] == 0 or x[1] == 0))

    single_electron_no_flip_kernel = ConstantKernel([0], single_electron_encoding)
    single_electron_brehm_kernel = GaussianKernel(
        single_electron_encoding,
        loc_model=basic_model_factory(single_electron_encoding, 1, final_layers=[ConstantScaleLayer(shift_loc, np.exp(shift_scale))]),
        scale_model=basic_model_factory(single_electron_encoding, ExponentialEncoding(1), final_layers=[ConstantScaleLayer(shift_scale)]),
    )
    single_electron_res_kernel = GaussianKernel(
        single_electron_encoding,
        loc_model=basic_model_factory(single_electron_encoding, 1, final_layers=[ConstantScaleLayer(shift_loc, np.exp(shift_scale))]),
        scale_model=basic_model_factory(single_electron_encoding, ExponentialEncoding(1), final_layers=[ConstantScaleLayer(shift_scale)]),
    )
    single_electron_eta_kernel = ConstantKernel([0], TrivialEncoding(4))
    single_electron_phi_kernel = ConstantKernel([0], TrivialEncoding(4))
    single_electron_pt_kernel = BranchingKernel.condition_on([
        single_electron_no_flip_kernel,
        single_electron_res_kernel,
        # single_electron_brehm_kernel,
    ], single_electron_flip_kernel) & single_electron_eta_kernel & single_electron_phi_kernel 

    dielectron_smear_kernel = single_electron_pt_kernel + single_electron_pt_kernel

    reordered_dielectron_smear_kernel = PermutationKernel(
        dielectron_smear_kernel,
        cond_permutation=[0, 2, 3, 4, 1,5, 6, 7],
    )

    # print(reordered_dielectron_smear_kernel.cond_dimension, event_flip_kernel.sample_dimension, event_flip_kernel.cond_dimension)
    # full_kernel = reordered_dielectron_smear_kernel | event_flip_kernel.cut(lambda x: (x.sum(dim=-1) > 0) and (x[0] == 0 or x[1] == 0))
    # print(full_kernel.draw(torch.zeros(10, 6)))

    # model = basic_model_factory(TrivialEncoding(2) & NopeEncoding(1)& TrivialEncoding(2) & NopeEncoding(3))
    model = basic_model_factory(CustomDielectronEncoding(TrivialEncoding(2) & NopeEncoding(1)& TrivialEncoding(2) & NopeEncoding(3)))
    swap_element = LambdaAction(
        input_dim=8,
        output_dim=1,
        input_fn=lambda x: torch.cat([x[:, 3:6], x[:, 0:3], x[:, 6:]], dim=-1),
    )
    group_action = FiniteGroupAction([swap_element], input_dim=8, output_dim=1)
    symmetrized_model = SymmetrizedModel(group_action, model)

    # Rewrite:
    # swap_element = LambdaAction(
    #     input_dim=8,
    #     output_dim=1,
    #     input_fn=lambda x: torch.cat([x[:, 3:6], x[:, 0:3], x[:, 6:]], dim=-1),
    # )

    # model = basic_model_factory(
    #     input =CustomDielectronEncoding(TrivialEncoding(2) & NopeEncoding(1)& TrivialEncoding(2) & NopeEncoding(3)), 
    #     symmetries = swap_element.to_group()
    # )

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=200,
        log_every_n_steps=1,
        logger=TensorBoardLogger(save_dir="iwpc_logs", name="di_electron"),
        num_sanity_val_steps=0,
    )

    # unlabelled = DielectronKernelTrainer(
    unlabelled = DielectronKernelTrainer.load_from_checkpoint(
        "iwpc_logs/di_electron/version_71/checkpoints/epoch=35-step=153864.ckpt",
        exact_kernel=exactly_one_flip_kernel,
        sampled_kernel=reordered_dielectron_smear_kernel,
        log_p_over_q_model=symmetrized_model,
        start_kernel_train_epoch = 0,
        start_discriminator_train_epoch = 0,
        kernel_opt_lr=1e-6,
        divergence=JensenShannonDivergence(), 
        target_cut_pass_prob=torch.tensor(0.002),
    )

    trainer.fit(
        unlabelled,
        datamodule=dm,
    )