# Investigating the Z-peak flip rate differences between the shifted MC and single electron truth models.
# Nov-2025

import torch
import numpy as np
import matplotlib.pyplot as plt
from TMTrain_single import KappaLightning as KappaMC
from TMTrain import KappaLightning as KappaTruth
from iwpc.utils import latest_ckpt

#Shifted MC
ckpt_path = latest_ckpt("/Users/albaburgosmondejar/QMISID/phi_logs/kappa/version_0")
module: KappaMC = KappaMC.load_from_checkpoint(ckpt_path, map_location="cpu")
module.eval() 

#Truth Logs
ckpt_path2 = latest_ckpt("/Users/albaburgosmondejar/QMISID/truth_logs/kappa/version_3")
module2: KappaTruth = KappaTruth.load_from_checkpoint(ckpt_path2, map_location="cpu")
module2.eval() 

def convert_to_prob(kappa: np.ndarray) -> np.ndarray:
    return 1-(1 / (1 + np.exp(-kappa)))

def convert_to_prob2(kappa: np.ndarray) -> np.ndarray:
    return (1 / (1 + np.exp(-kappa)))

@torch.no_grad()
def evaluate_for_visualiser(z: np.ndarray, model: torch.nn.Module = module) -> dict:
    z_proc = np.stack([
        z[:, 0] / z[:, 1],
        z[:, 2],
    ], axis=1).astype(np.float32)
    z_tensor = torch.from_numpy(z_proc)
    output = model(z_tensor) 
    kappa = output[:, 0].detach().cpu().numpy()
    prob = convert_to_prob(kappa)             

    return prob

@torch.no_grad()
def evaluate_for_visualiser2(z: np.ndarray, model: torch.nn.Module = module2) -> dict:
    z_proc = np.stack([
        z[:, 0] / z[:, 1],
        z[:, 2],
    ], axis=1).astype(np.float32)
    z_tensor = torch.from_numpy(z_proc)
    output = model(z_tensor) 
    kappa = output[:, 0].detach().cpu().numpy()
    prob = convert_to_prob2(kappa)             
    return prob

pt_vals = np.linspace(5e3, 260000, 500)   
eta_vals = np.linspace(0.0, 2.6, 500)  

ETA, PT = np.meshgrid(eta_vals, pt_vals)
charge = -1.0 

z_input = np.column_stack([
    np.full(PT.size, charge),
    PT.ravel(),
    ETA.ravel()
]).astype(np.float32)

result_1 = evaluate_for_visualiser(z_input, module)
result_2 = evaluate_for_visualiser2(z_input, module2)

diff = result_2-result_1
diff_grid = diff.reshape(PT.shape)

plt.figure(figsize=(8, 6))
plt.pcolormesh(ETA, PT, abs(diff_grid), shading="auto", cmap="magma").set_mouseover(True)
plt.colorbar(label="Prob_MC - Prob_Truth")

plt.xlabel(r"$\eta$")
plt.ylabel("Pt")
plt.title("Flip Rate Difference between MC and Single Electron Truth")
plt.grid(False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(diff, bins=100, alpha=0.7, color="gray", edgecolor="black")
plt.xlabel("Prob_MC - Prob_Truth")
plt.ylabel("Count")
plt.title("Distribution of Flip Rate Differences")
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Comparison To Likelihood Method

theta_values = np.array([
    0.00013603419438368292, 0.00041625970148917446, 0.001951453451835894,
    0.006737350203021686, 0.000442155365513619, 0.0013640946105646368,
    0.006273724025009253, 0.019813836196195922, 0.0007449079882202447,
    0.002060373346909339, 0.009381704705272664, 0.030125325889850618,
    0.0012004459238947884, 0.003391764660073182, 0.01467007756299532,
    0.0426039933961484
])

eta_pt_pairs = np.array([
    [0.685, 35.0],
    [1.445, 35.0],
    [1.76,  35.0],
    [2.3,   35.0],
    [0.685, 75.0],
    [1.445, 75.0],
    [1.76,  75.0],
    [2.3,   75.0],
    [0.685, 150.0],
    [1.445, 150.0],
    [1.76,  150.0],
    [2.3,   150.0],
    [0.685, 1400.0],
    [1.445, 1400.0],
    [1.76,  1400.0],
    [2.3,   1400.0],
])


charge = -1.0
z_input = np.column_stack([
    np.full(len(eta_pt_pairs), charge),
    eta_pt_pairs[:, 1]*1e3,  
    eta_pt_pairs[:, 0]
]).astype(np.float32)

like_probs = evaluate_for_visualiser2(z_input, module2)
diff_like = like_probs - theta_values

plt.figure(figsize=(6, 4))

plt.hist(
    diff_like,
    bins=60,
    alpha=0.6,
    edgecolor='black',
    density=True,
    label='Likelihood'
)

plt.hist(
    diff,
    bins=60,
    alpha=0.6,
    edgecolor='red',
    density=True,
    label='Neural Network'
)

plt.xlabel("Data-Monte Carlo")
plt.ylabel("Density")
plt.title("Distribution of Flip Rate Differences (Normalized)")
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

