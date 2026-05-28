# Experimentation with cut kernels, which are kernels that are only defined on a subset of the input space.
# FiniteCutKernel interface is used in final Dielectron.py script

from pathlib import Path

import numpy as np
import torch

from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.learn_dist.kernels.constant_kernel import ConstantKernel
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.kernels.fixed_finite_kernel import FixedFiniteKernel
from iwpc.learn_dist.kernels.mixture_kernel import MixtureKernel

eta_pt_encoding = TrivialEncoding(2)

def load_benchmark_mc_inputs(
    num_rows: int = 100000,
    dataset_dir: Path = Path("/Users/albaburgosmondejar/Desktop/BenchmarkMC"),
) -> tuple[torch.Tensor, torch.Tensor]:
    import pandas as pd
    
    pkl_files = sorted(dataset_dir.glob("*.pkl"))
    benchmark_file = min(pkl_files, key=lambda p: p.stat().st_size)
    df = pd.read_pickle(benchmark_file)

    df = df.iloc[:num_rows]

    l1_eta = df["l1_eta"].to_numpy(dtype=np.float32)
    l2_eta = df["l2_eta"].to_numpy(dtype=np.float32)
    l1_pt = (1.0 / df["l1_q_over_pt"].abs()).to_numpy(dtype=np.float32)
    l2_pt = (1.0 / df["l2_q_over_pt"].abs()).to_numpy(dtype=np.float32)

    two_electron = torch.from_numpy(
        np.stack([l1_pt, l1_eta, l2_pt, l2_eta], axis=1)
    )
    return two_electron


random_two_electron = load_benchmark_mc_inputs()

eta_pt_encoding = TrivialEncoding(2)

trident_flip_pt_err_kernel = GaussianKernel(eta_pt_encoding)
resolution_flip_pt_err_kernel = GaussianKernel(eta_pt_encoding)
flipped_pt_err_kernel = MixtureKernel([trident_flip_pt_err_kernel, resolution_flip_pt_err_kernel], eta_pt_encoding)
flipped_kinematics_err_kernel = ConstantKernel([0.], eta_pt_encoding) & flipped_pt_err_kernel

is_flipped_kernel = FiniteKernel(2, eta_pt_encoding)

unflipped_kinematics_err_kernel = ConstantKernel([0.], eta_pt_encoding) & GaussianKernel(eta_pt_encoding)
single_electron_det_resp = [unflipped_kinematics_err_kernel, flipped_kinematics_err_kernel] | is_flipped_kernel

# Our actual model of the detector response
two_electron_det_response = single_electron_det_resp + single_electron_det_resp

two_electron_is_flipped_kernel = is_flipped_kernel + is_flipped_kernel
cut_two_electron_is_flipped_kernel = two_electron_is_flipped_kernel.cut(
    lambda x: x.equal(torch.tensor([0, 1])) or x.equal(torch.tensor([1, 0])),
)

# Form of the cut detector response we will have to use for the DivGradKernel trainer for compatibility
cut_two_electron_det_response = [
    unflipped_kinematics_err_kernel + flipped_kinematics_err_kernel,
    flipped_kinematics_err_kernel + unflipped_kinematics_err_kernel
] | cut_two_electron_is_flipped_kernel

print(two_electron_det_response.draw(random_two_electron))
print(cut_two_electron_det_response.draw(random_two_electron))