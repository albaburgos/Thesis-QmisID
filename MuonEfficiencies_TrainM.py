
# Experimentation: Muon Efficiencies with Multi-Class Cross-Entropy Loss for muon efficiencies.

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

class KappaLightning(L.LightningModule, Visualisable):
    def __init__(self, lr: float = 4e-3):
        super().__init__()
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        input_encoding  = TrivialEncoding(2) & ContinuousPeriodicEncoding((-torch.pi, torch.pi))
        #input_encoding  = TrivialEncoding(2)
        target_encoding = TrivialEncoding(4)

        self.log_id_model = basic_model_factory(input_encoding, 1)
        self.log_ms_model = basic_model_factory(input_encoding, 1)

        self.save_hyperparameters()
    
    def construct_logit_full_dist(self, x):
        id_probs = torch.sigmoid(self.log_id_model(x))[:, 0]
        ms_probs = torch.sigmoid(self.log_ms_model(x))[:, 0]

        probs = torch.stack([
            (1 - id_probs) * (1 - ms_probs),
            (1 - id_probs) * ms_probs,
            id_probs * (1 - ms_probs),
            id_probs * ms_probs,
        ], dim=1)
        probs = probs.clamp_min(1e-18)
        return torch.log(probs)
    
    def _shared_step(self, batch, stage: str):
        x, y, w = batch
           
        y_int = (y[:, 0].long() << 1) + y[:, 1].long()
        logit = self.construct_logit_full_dist(x)
        loss = (self.criterion(logit, y_int) * w).mean()

        #preds = torch.argmax(logs, dim=1)
        #acc = (preds == y_int).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"),
                 on_step=True, on_epoch=True, batch_size=x.size(0))
        #self.log(f"{stage}_acc", acc, prog_bar=True,
                 #on_step=True, on_epoch=True, batch_size=x.size(0))

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
            threshold=0,
            threshold_mode="rel",
            cooldown=2,
            # min_lr=1e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def get_input_scalars(self):
        return [
            Scalar("charge", bins=np.linspace(-1,1,2)),
            Scalar("Pt", bins=np.linspace(5e3, 500e3, 100)),
            Scalar("eta", bins=np.linspace(-2.5,2.5,100)),
            Scalar("phi", bins=np.linspace(-np.pi,np.pi,100)),
        ]
    
    def get_output_scalars(self):
        return [
            ScalarFunction(lambda x: x["eID"], "eID", bins=np.linspace(0, 1, 100)),
            ScalarFunction(lambda x: x["eMS"], "eMS", bins=np.linspace(0, 1, 100)),
        ]

    def evaluate_for_visualiser(self, z):
        self.eval()
        self.cpu()

        z = np.stack([
            z[:, 0]/z[:, 1],
            z[:, 2],
            z[:, 3]
        ], axis=1)

        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logit = self.construct_logit_full_dist(z)
            probs = torch.softmax(logit, dim=1)
            probs = probs.detach().cpu().numpy()
        return { "eID": (probs[:,3]+probs[:,2])/(probs[:,3]+probs[:,1]+probs[:,2]+probs[:,0]), "eMS": (probs[:,3]+probs[:,1])/(probs[:,3]+probs[:,1]+probs[:,2]+probs[:,0])}
    
    @property
    def center_point(self):
        return [-1, 45e3, 0.01, 0.01]

if __name__ == "__main__":
    from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule

    path = "/Users/albaburgosmondejar/Desktop/Muon/"

    dm = PandasDirDataModule(
        dataset_dir=path,
        feature_cols=["q_over_pt", "eta", "phi"],
        target_cols=["has_id_track", "has_ms_track"],
        #weight_col="weight",
        split=0.8,
        dataloader_kwargs={"num_workers": 8, "batch_size": 2**15},
    )

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    #wrap model in LightningModule
    lit_model = KappaLightning(lr=3e-3)

    trainer = L.Trainer(
        accelerator="mps",
        # precision="16-mixed",
        max_epochs=20000,
        log_every_n_steps=200,
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min"),
            LearningRateMonitor(),
            # EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
        logger=TensorBoardLogger(save_dir="muonm_logs", name="kappa"),
        num_sanity_val_steps=0,
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
