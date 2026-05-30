# QmisID — Electron Charge Misidentification for the ATLAS di-Higgs analysis

Code accompanying the thesis on learning electron charge misidentification (QmisID) probabilities using neural networks and kernel-based methods. The project covers single-electron network training (Sections 3.1, 3.2) and an unlabelled dielectron approach with in-situ detector calibration (Section 3.3), built on the [IWPC](https://github.com/) framework with PyTorch Lightning.

---

## Trained models

Pre-trained models are provided in ONNX format and can be loaded with any ONNX-compatible runtime:

| File | Description |
|---|---|
| `qmisid.onnx` | Single-electron QmisID network trained on simulation |
| `qmisid_data.onnx` | Dielectron-calibrated QmisID network trained on data |

---

## Main training scripts

### `TMTrain.py` — Single-electron network (Sections 3.1, 3.2)

Trains a neural network to predict the per-electron charge misidentification probability κ directly from simulation truth labels. Uses the IWPC `PandasDirDataModule` and a `KappaLightning` module with TensorBoard logging and early stopping.

```bash
python TMTrain.py
```

### `DielectronUnlabelled.py` — Dielectron calibration (Section 3.3)

Trains an unlabelled kernel-based model using dielectron pairs, without requiring per-electron truth labels. Applies an f-divergence minimisation (`FDivergenceMinimizingKernelTrainer`) with a mixture kernel over the dielectron invariant mass, enabling in-situ calibration to data.

```bash
python DielectronUnlabelled.py
```

---

## Repository structure

### Experimentation — Sections 3.1,3.2 (`TMTrain*`)

Variants on the single-electron network:

| Script | Description |
|---|---|
| `TMTrain_single.py` | Single-electron dataset training |
| `TMTrain_phi.py` | Training with φ-dependent encoding |
| `TMTrain_KappaNet.py` | Alternative network architecture |
| `TMTrain_Z_peak.py` | Investigating the Z-peak shift |
| `TMTrain_onnx.py` | Training with inline ONNX export |
| `TMTrain_Likelihood_script.py` | Likelihood ratio evaluation script |

### Experimentation — Section 3.3 (`Dielectron*`, `gaussian_kernel*`)

Variants on the dielectron unlabelled approach and kernel design:

| Script | Description |
|---|---|
| `DielectronExp.py` | Experimental dielectron training |
| `DielectronDummy.py`, `DielectronDummyTrain.py` | Toy Dielectron setups |
| `DielectronUnlabelled1deta.py`, `DielectronUnlabelled1dpt.py` | 1D Dielectron training in η and pT |
| `Dielectron_DiscreteCut.py` | Discrete cut-based dielectron baseline |
| `Dielectron_add_cond_kernel.py` | Conditional kernel addition |
| `Dielectron_binned_divergence.py` | Binned f-divergence evaluation utility|
| `Dielectron_invariant_mass_plots.py` | Invariant mass distribution plots |

| `gaussian_kernel_single.py` | Single Gaussian kernel |
| `gaussian_kernel_global_alpha.py` | Global α parameter training |
| `gaussian_kernel_variable_alpha.py` | Variable α parameter training |
| `gaussian_kernel_mixture_eval.py` | Mixture kernel evaluation |
| `gaussian_kernel_two_trainers.py` | Two-trainer alternating optimisation |
| `gaussian_kernel_smearing.py` | pt error smearing plots |
| `gaussian_kernel_plot.py`, `gaussian_kernel_plotting.py` | Kernel plots |

### Closure tests (`closure_*`)

Validate that the trained networks reproduce the correct misidentification rate on held-out samples:

| Script | Description |
|---|---|
| `closure_single_electron.py` | Closure on single-electron simulation (Sections 3.1,3.2) |
| `closure_dielectron.py` | Closure on dielectron pairs (Section 3.3)) |
| `closure_onnx_model.py` | Closure evaluation from exported ONNX model |
| `closure_benchmark_MC.py` | MC benchmark comparison |
| `closure_benchmark_MC_allpanels.py` | MC benchmark with full panel layout |

### Visualisation (`vis*`)

Interactive IWPC visualiser scripts for inspecting learned functions:

| Script | Description |
|---|---|
| `vis.py` | Single-electron QmisID probability visualiser |
| `vis_diunlabelled.py` | Dielectron kernel model visualiser |
| `vis_unlabelled.py` | Unlabelled kernel model visualiser |
| `vis_kernel.py` | Kernel function visualiser |

### Muon efficiency experiments (`MuonEfficiencies_*`)

Experiments with Muon Efficiencies.

| Script | Description |
|---|---|
| `MuonEfficiencies_TrainB.py` | Muon efficiency training — binary cross-entropy |
| `MuonEfficiencies_TrainM.py` | Muon efficiency training — multi-class cross-entropy |

### Utilities (`_*`)

Custom encoding and data preparation scripts:

| Script | Description |
|---|---|
| `_InvariantMass.py` | Custom dielectron encoding (`CustomDielectronEncoding`) |
| `_make_data.py` | Data prep script |
| `_make_data_original.py` | Original data prep script provided by Cambridge ATLAS group |
| `_export_onnx.py` | ONNX export utility |

---

## Dependencies

- [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- [IWPC](https://github.com/) — internal framework providing data modules, kernel trainers, encodings, and visualisers
- NumPy, Matplotlib, Bokeh (visualisation)
