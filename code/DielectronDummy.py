# Experimentation: Attempt at constructing dielectron kernel model

from pathlib import Path

import numpy as np
import torch

from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.encodings.nope_encoding import NopeEncoding
from iwpc.learn_dist.kernels.constant_kernel import ConstantKernel
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.kernels.mixture_kernel import MixtureKernel
from iwpc.learn_dist.kernels.branching_kernel import BranchingKernel
from iwpc.learn_dist.kernels.add_cond_kernel import AddCondKernel
from iwpc.learn_dist.kernels.finite_kernel import FiniteKernel
from iwpc.learn_dist.kernels.restructuring_kernel import RestructuringKernel


dummy_electron =torch.rand(size=(10,3))
dummy_dielectron =torch.rand(size=(10,6))

kinematics_encoding = TrivialEncoding(1) & NopeEncoding(1) & TrivialEncoding(1)

eta_phi_err_kernel = ConstantKernel([0,0], kinematics_encoding)

br_q_over_pt_err_kernel = GaussianKernel(kinematics_encoding)
res_q_over_pt_err_kernel = GaussianKernel(kinematics_encoding)

flipped_err_kernel = MixtureKernel([eta_phi_err_kernel & br_q_over_pt_err_kernel, eta_phi_err_kernel & res_q_over_pt_err_kernel], kinematics_encoding)
unflipped_err_kernel = ConstantKernel([0,0,0], kinematics_encoding)
smearing_kernel = BranchingKernel(
    [(unflipped_err_kernel), (flipped_err_kernel)],
    [0,],
    lambda x: x[:, 0]
)
charge_flip_kernel = FiniteKernel(2, kinematics_encoding)

single_electron_kernel = smearing_kernel | charge_flip_kernel
dielectron_kernel = single_electron_kernel + single_electron_kernel

reordered_kernel = RestructuringKernel(smearing_kernel + smearing_kernel, [0,2,3,4,1,5,6,7], 8)
dielectron_flip_kernel = charge_flip_kernel + charge_flip_kernel
cut_dielectron_flip_kernel = dielectron_flip_kernel.cut(lambda x: x.sum() == 1)

dummy_dielectron_with_flip_decision = torch.concat([cut_dielectron_flip_kernel.draw(dummy_dielectron), dummy_dielectron], dim=1)
print(reordered_kernel.draw(dummy_dielectron_with_flip_decision))
