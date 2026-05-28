# Single Electron Training Script

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

def do_nothing(x):
    return x['kappa']

def convert_to_prob(x):
    return 1 / (1 + np.exp(-x['kappa']))

class KappaLightning(L.LightningModule, Visualisable):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

        input_encoding  = TrivialEncoding(2) & ContinuousPeriodicEncoding((-torch.pi, torch.pi))
        #input_encoding  = TrivialEncoding(2)
        target_encoding = TrivialEncoding(1)
        self.model = basic_model_factory(input_encoding,target_encoding) 

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y, _ = batch

        logits = self.forward(x)
        y = y.view_as(logits)
        loss = self.criterion(logits, y)

        probs = torch.sigmoid(logits)
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
            ScalarFunction(do_nothing, "kappa"),
            ScalarFunction(convert_to_prob, "flip_prob", bins=np.linspace(0,0.1,100)),
        ]

    def evaluate_for_visualiser(self, x):
        self.eval()
        self.cpu()

        x = np.stack([
            x[:, 0] / x[:, 1],
            x[:, 2],
            x[:, 3],
        ], axis=1)
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = self.forward(x).cpu().numpy()[:, 0]

        return {"kappa": output}
    
    @property
    def centre_point(self):
        return [-1, 45e3, 0.5, 0.0]
    

if __name__ == "__main__":
    from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule

    path = "/Users/albaburgosmondejar/Desktop/shuhui_truth_dm"
    import os
    os.listdir("/Users/albaburgosmondejar/Desktop/shuhui_truth_dm") 

    dm = PandasDirDataModule(
        dataset_dir=path,
        feature_cols=["truth_el_q_over_pt", "truth_el_eta", "truth_el_phi"],
        target_cols=["is_flipped"],
        split=0.8,
        dataloader_kwargs={"num_workers": 8, "batch_size": 2**15},
    )
    print(dm.file_sizes)

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
        logger=TensorBoardLogger(save_dir="phi_training", name="kappa"),
        num_sanity_val_steps=0,
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)