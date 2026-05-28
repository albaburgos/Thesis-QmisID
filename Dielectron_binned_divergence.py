# Skeleton code to calculate binned divergence estimates
# Used for debugging Dielectron Kernel training

from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.scalars.scalar_function import ScalarFunction
from iwpc.accumulators.histogram_accumulator import HistogramAccumulator
from iwpc.accumulators.binned_weighted_stat_accumulator import BinnedWeightedStatAccumulator
from tqdm import tqdm
import numpy as np

dm = PandasDirDataModule(
        dataset_dir="/Users/albaburgosmondejar/Desktop/BenchmarkMC",
        feature_spec=[
            ["l1_pt", "l1_eta", "l1_phi", "l2_pt", "l2_eta", "l2_phi"],  # base: OC kinematics (6D), fed into exact_kernel
            ["l1_pt", "l1_eta","l1_phi", "l2_pt", "l2_eta", "l2_phi", "b", "b"], # data: SC observables (8D), fed into discriminator
            'opposite_charge',
        ],
        weight_col="weight",
        split=0.8,
        dataloader_kwargs={"num_workers": 8, "batch_size": 2**17},
        # use_in_memory_dataset=True,
    )

scalars = [ScalarFunction(lambda df: df['l1_pt'].values, 'pt1', bins=np.linspace(0, 150e3))]

def calculate_weight_sum(scalars, dm: PandasDirDataModule, label):
    weight_sum_hist = HistogramAccumulator([s.bins for s in scalars])
    for _, df in tqdm(dm.file_iter(include_train_files=False), total = dm.num_validation_files):
        df = df.loc[df['opposite_charge'] == label]
        weight_sum_hist.update([s(df) for s in scalars], df['weight'].values)
    
    return weight_sum_hist

p_hist = calculate_weight_sum(scalars, dm, 1)
q_hist = calculate_weight_sum(scalars, dm, 0)


def calculate_binned_cross_entropy(scalars, dm: PandasDirDataModule, marginalised_log_p_over_q):
    p_loss_acc = BinnedWeightedStatAccumulator([s.bins for s in scalars])
    q_loss_acc = BinnedWeightedStatAccumulator([s.bins for s in scalars])
    for _, df in tqdm(dm.file_iter(include_train_files=False), total = dm.num_validation_files):
        df = df.loc[df['opposite_charge'] == label]
        log_p_over_q = model() - marginalised_log_p_over_q[idxs]

        p_df = df.loc[df['label'].values == 0]
        q_df = df.loc[df['label'].values == 1]
        p_loss_acc.update([s(p_df) for s in scalars], np.log(np.sigmoid(log_p_over_q[df['label'].values == 0])), weights=)
        q_loss_acc.update([s(q_df) for s in scalars], np.log(np.sigmoid(-log_p_over_q[df['label'].values == 1])), weights=)
    
    return p_loss_acc.weighted_mean_hist - q_loss_acc.weighted_mean_hist


print()
