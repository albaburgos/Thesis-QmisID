#Experimentation: Muon Efficiencies with Binary Cross-entropy loss for ID and MS efficiencies

import os
import math
import datetime
from pathlib import Path
from typing import Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from iwpc.calculate_divergence import DivergenceResult
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.data_modules.pandas_directory_data_module_builder import PandasDirDataModuleBuilder
from iwpc.models.utils import basic_model_factory
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.encodings.continuous_periodic_encoding import ContinuousPeriodicEncoding
from tqdm import tqdm
from iwpc.visualise.multidimensional_function_visualiser import MultidimensionalFunctionVisualiser
import matplotlib.pyplot as plt
from iwpc.visualise.visualisable import Visualisable
from iwpc.scalars.scalar import Scalar
from iwpc.scalars.scalar_function import ScalarFunction
from iwpc.models.layers import ConstantScaleLayer
from torch.optim.lr_scheduler import LambdaLR

def do_nothing_kappa1(x):
    return x["kappa1"]

def do_nothing_kappa2(x):
    return x["kappa2"]

def convert_to_prob_kappa1(x):
    k = x["kappa1"]
    return 1.0 - (1.0 / (1.0 + np.exp(-k)))

def convert_to_prob_kappa2(x):
    k = x["kappa2"]
    return 1.0 - (1.0 / (1.0 + np.exp(-k)))

def inv_sigmoid(x): 
    return np.log(x)-np.log(1-x)

def f_phi(phi1, phi2):
    p = 1 / (1 + F.softplus(-phi1) + F.softplus(-phi2) + F.softplus(-phi1 - phi2))
    return torch.logit(p, eps=1e-6)

class KappaLightning(L.LightningModule, Visualisable):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        input_encoding  = TrivialEncoding(2) & ContinuousPeriodicEncoding((-torch.pi, torch.pi))
        #input_encoding  = TrivialEncoding(2)
        target_encoding = TrivialEncoding(2)
        self.model = basic_model_factory(
            input_encoding,
            target_encoding,
            final_layers=[ConstantScaleLayer(shift=inv_sigmoid(0.95))],
        )

        self.save_hyperparameters()

    def forward(self, x):
        out = self.model(x) 
        if isinstance(out, dict):
            out = out["kappa"]
        phi1 = out[:, 0:1]                
        phi2 = out[:, 1:2]               
        return phi1, phi2
    
    def _shared_step(self, batch, stage: str):
        x, y, w = batch
           
        y = y.bool().all(dim=1).float().unsqueeze(1)
        phi1, phi2 = self.forward(x)
        f = f_phi(phi1, phi2)  
        y = y.view_as(phi1)
        loss = (self.criterion(f, y) * w).mean()

        probs = torch.sigmoid(f)
        preds = (probs > 0.5).float()
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"),
                 on_step=True, on_epoch=True, batch_size=x.size(0))
        self.log(f"{stage}_acc", acc, prog_bar=True,
                 on_step=True, on_epoch=True, batch_size=x.size(0))

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=self.lr,
        total_steps=self.trainer.estimated_stepping_batches,
        pct_start=0.03,
        anneal_strategy="cos",
    )
        return [optimizer], [scheduler]
    
    def get_input_scalars(self):
        return [
            Scalar("charge", bins=np.linspace(-1,1,2)),
            Scalar("Pt", bins=np.linspace(5e3, 500e3, 100)),
            Scalar("eta", bins=np.linspace(0,2.5,100)),
            Scalar("phi", bins=np.linspace(-np.pi,np.pi,100)),
        ]
    
    def get_output_scalars(self):
        return [
            ScalarFunction(do_nothing_kappa1, "kappa1"),
            ScalarFunction(convert_to_prob_kappa1, "flip_prob1", bins=np.linspace(0,0.1,100)),
            ScalarFunction(do_nothing_kappa2, "kappa2"),
            ScalarFunction(convert_to_prob_kappa2, "flip_prob2", bins=np.linspace(0,0.1,100)),
        ]

    def evaluate_for_visualiser(self, z):
        self.eval()
        self.cpu()

        z = np.stack([
            z[:, 0]/z[:, 1],
            z[:, 2],
            z[:, 3],
        ], axis=1)

        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        with torch.no_grad():

            out = self.model(z)
            if isinstance(out, dict):
                out = out["kappa"]
            out = out.detach().cpu().numpy()   

        return {"kappa1" : out[:, 0], "kappa2": out[:, 1]}
    
    #@property
    #def center_point(self):
       # return [-1, 45e3, 0.01]

if __name__ == "__main__":
    from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule

    path = "/Users/albaburgosmondejar/Desktop/Muon/"

    dm = PandasDirDataModule(
        dataset_dir=path,
        feature_cols=["q_over_pt", "eta", "phi"],
        target_cols=["has_id_track", "has_ms_track"],
        #weight_col="ububhn",
        split=0.8,
        dataloader_kwargs={"num_workers": 8, "batch_size": 2**15},
    )

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    #wrap model in LightningModule
    lit_model = KappaLightning(lr=1e-3)

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=200,
        log_every_n_steps=50,
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min"),
            # EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            
        ],
        logger=TensorBoardLogger(save_dir="muon_logs", name="kappa"),
        num_sanity_val_steps=0,
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)