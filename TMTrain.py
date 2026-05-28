# Main Training Script

import os
import math
import datetime
from pathlib import Path
from typing import Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def do_nothing(x):
    return x['kappa']

def convert_to_prob(x):
    return 1- ( 1 / (1 + np.exp(-x['kappa'])))

def inv_sigmoid(x): 
    return np.log(x)-np.log(1-x)

def f_phi(phi1, phi2):
    term1 = F.softplus(phi1 + phi2)
    stacked = torch.stack([phi1, phi2], dim=-1)
    term2 = torch.logsumexp(stacked, dim=-1)
    return term1 - term2

class KappaLightning(L.LightningModule, Visualisable):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        #input_encoding  = TrivialEncoding(2) & ContinuousPeriodicEncoding((-torch.pi, torch.pi))
        input_encoding  = TrivialEncoding(2)
        target_encoding = TrivialEncoding(1)
        self.model = basic_model_factory(
            input_encoding,
            target_encoding,
            final_layers=[ConstantScaleLayer(shift=inv_sigmoid(0.95))],
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y, w = batch

        x1 = x[:, [0,2]]
        x2 = x[:, [1,3]]

        logits1 = self.forward(x1)
        logits2 = self.forward(x2)
        f = f_phi(logits1, logits2)
        y = y.view_as(logits1)
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
            #Scalar("phi", bins=np.linspace(-np.pi,np.pi,100)),
        ]
    
    def get_output_scalars(self):
        return [
            ScalarFunction(do_nothing, "kappa"),
            ScalarFunction(convert_to_prob, "flip_prob", bins=np.linspace(0,0.1,100)),
        ]

    def evaluate_for_visualiser(self, z):
        self.eval()
        self.cpu()

        z = np.stack([
            z[:, 0] / z[:, 1],
            z[:, 2],
        ], axis=1)

        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        with torch.no_grad():

            output = self.forward(z).cpu().numpy()[:, 0]

        return {"kappa": output}
    
    @property
    def center_point(self):
        return [-1, 45e3, 0.01]

if __name__ == "__main__":
    from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule

    path = "/Users/albaburgosmondejar/Desktop/Dataset4/"

    dm = PandasDirDataModule(
        dataset_dir=path,
        feature_cols=["l1_q_over_pt", "l2_q_over_pt", "l1_eta", "l2_eta"],
        target_cols=["label"],
        weight_col="weight",
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
        logger=TensorBoardLogger(save_dir="shift_logs", name="kappa"),
        num_sanity_val_steps=0,
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
