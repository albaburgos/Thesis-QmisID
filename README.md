# QmisID — Electron Charge Misidentification with Machine Learning

Code accompanying the thesis on learning electron charge misidentification (QmisID) probabilities using neural networks and kernel-based methods. The project covers single-electron network training (Chapters 3–4) and an unlabelled dielectron approach with in-situ detector calibration (Chapter 5), built on the [IWPC](https://github.com/) framework with PyTorch Lightning.

---

## Trained models

Pre-trained models are provided in ONNX format and can be loaded with any ONNX-compatible runtime:

| File | Description |
|---|---|
| `qmisid.onnx` | Single-electron QmisID network trained on simulation |
| `qmisid_data.onnx` | Dielectron-calibrated QmisID network trained on data |

---

## Main training scripts

### `TMTrain.py` — Single-electron network (Chapters 3–4)

Trains a neural network to predict the per-electron charge misidentification probability κ directly from simulation truth labels. Uses the IWPC `PandasDirDataModule` and a `KappaLightning` module with TensorBoard logging and early stopping.

```bash
python TMTrain.py
```

### `DielectronUnlabelled.py` — Dielectron calibration (Chapter 5)

Trains an unlabelled kernel-based model using dielectron pairs, without requiring per-electron truth labels. Applies an f-divergence minimisation (`FDivergenceMinimizingKernelTrainer`) with a mixture kernel over the dielectron invariant mass, enabling in-situ calibration to data.

```bash
python DielectronUnlabelled.py
```

---

## Repository structure

### Experimentation — Chapters 3–4 (`TMTrain*`)

Variants and ablations on the single-electron network:

| Script | Description |
|---|---|
| `TMTrain_single.py` | Minimal single-variable training |
| `TMTrain_phi.py` | Training with φ-dependent encoding |
| `TMTrain_KappaNet.py` | Alternative network architecture |
| `TMTrain_Z_peak.py` | Training with Z-peak event selection |
| `TMTrain_onnx.py` | Training with inline ONNX export |
| `TMTrain_Likelihood_script.py` | Likelihood ratio evaluation script |

### Experimentation — Chapter 5 (`Dielectron*`, `gaussian_kernel*`)

Variants on the dielectron unlabelled approach and kernel design:

| Script | Description |
|---|---|
| `DielectronExp.py`, `DielectronExp2.py`, `DielectronExp3.py` | Experimental dielectron training variants |
| `DielectronDummy.py`, `DielectronDummyTrain.py` | Toy / dummy dielectron setups |
| `DielectronUnlabelled1deta.py`, `DielectronUnlabelled1dpt.py` | 1D sweeps over η and pT |
| `Dielectron_DiscreteCut.py` | Discrete cut-based dielectron baseline |
| `Dielectron_add_cond_kernel.py` | Conditional kernel addition |
| `Dielectron_binned_divergence.py` | Binned f-divergence evaluation |
| `Dielectron_invariant_mass_plots.py` | Invariant mass distribution plots |
| `gaussian_kernel_single.py` | Single Gaussian kernel baseline |
| `gaussian_kernel_smearing.py` | Smearing study with Gaussian kernel |
| `gaussian_kernel_global_alpha.py` | Global α parameter training |
| `gaussian_kernel_variable_alpha.py` | Per-bin variable α training |
| `gaussian_kernel_mixture_eval.py` | Mixture kernel evaluation |
| `gaussian_kernel_two_trainers.py` | Two-trainer alternating optimisation |
| `gaussian_kernel_plot.py`, `gaussian_kernel_plotting.py` | Kernel diagnostic plots |

### Closure tests (`closure_*`)

Validate that the trained networks reproduce the correct misidentification rate on held-out samples:

| Script | Description |
|---|---|
| `closure_single_electron.py` | Closure on single-electron simulation |
| `closure_dielectron.py` | Closure on dielectron pairs |
| `closure_onnx_model.py` | Closure evaluation from exported ONNX model |
| `closure_benchmark_MC.py` | MC benchmark comparison |
| `closure_benchmark_MC_allpanels.py` | MC benchmark with full panel layout |

### Visualisation (`vis*`)

Interactive IWPC visualiser scripts for inspecting learned functions:

| Script | Description |
|---|---|
| `vis.py` | Single-electron QmisID probability visualiser |
| `vis_unlabelled.py` | Unlabelled kernel model visualiser |
| `vis_diunlabelled.py` | Dielectron unlabelled model visualiser |
| `vis_kernel.py` | Kernel function visualiser |

### Muon efficiency experiments (`MuonEfficiencies_*`)

Exploratory adaptation of the QmisID approach to muon efficiency estimation:

| Script | Description |
|---|---|
| `MuonEfficiencies_TrainB.py` | Muon efficiency training — background model |
| `MuonEfficiencies_TrainM.py` | Muon efficiency training — signal model |

### Utilities (`_*`)

Custom encoding and data preparation scripts:

| Script | Description |
|---|---|
| `_InvariantMass.py` | Custom dielectron encoding (`CustomDielectronEncoding`) |
| `_make_data.py` | Data preparation pipeline |
| `_make_data_original.py` | Original data preparation (reference) |
| `_export_onnx.py` | ONNX export utility |

---

## Dependencies

- [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- [IWPC](https://github.com/) — internal framework providing data modules, kernel trainers, encodings, and visualisers
- [ONNX Runtime](https://onnxruntime.ai/) for model inference
- NumPy, Matplotlib, Bokeh (visualisation)
