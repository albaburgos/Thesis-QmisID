# Experimentation: Pytorch lightning module Kappanet for single-electron QmisID training
# Pytorch lightning applied to /Users/albaburgosmondejar/Desktop/shuhui_truth_dm/file_0.pkl Dataset
# Oct-2025

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, Trainer, seed_everything, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd

DATA_PATH = "/Users/albaburgosmondejar/Desktop/shuhui_truth_dm/file_0.pkl"  
BATCH_SIZE = int(50000)
VAL_FRAC = 0.2
MAX_EPOCHS = 5
LR = 1e-4
HIDDEN = 64
NUM_WORKERS = 16 # to load data simultaneously
SEED = 42
DEVICE = "gpu"

class KappaNet(nn.Module): #Three linear maps, 2 non-linear activation ReLU
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, e):
        return self.net(e).squeeze(-1)

def f_phi(phi1, phi2):
    term1 = F.softplus(phi1 + phi2)
    stacked = torch.stack([phi1, phi2], dim=-1)
    term2 = torch.logsumexp(stacked, dim=-1)
    return term1 - term2

class KappaLightning(LightningModule):
    def __init__(self, input_dim, hidden_dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters() #input dim, hidden dim, lr saved automatically in self
        self.model = KappaNet(input_dim=input_dim, hidden_dim=hidden_dim)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, e): # Run model on input tensor, what happens to the output tensor
        return self.model(e)

    def _pairwise_loss(self, e1, e2, labels):
        phi1, phi2 = self(e1), self(e2)
        f = f_phi(phi1, phi2)
        return self.criterion(f, labels.float())

    def training_step(self, batch, batch_idx):
        e1, e2, labels = batch # Tuple of data + labels
        loss = self._pairwise_loss(e1, e2, labels) # Calculate cross-entropy loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True) 
        return loss
    
    def validation_step(self, batch, batch_idx):
        e1, e2, labels = batch
        loss = self._pairwise_loss(e1, e2, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self): #pass model parameters and the learning rate
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr) #hparams used to rank runs


def main():
    seed_everything(SEED) #reproducibility

    logger = TensorBoardLogger(save_dir="lightning_logs", name="kappa_real")

    df = pd.read_pickle(DATA_PATH)

    feat_cols = [
        "truth_el_pt",
        "truth_el_eta",
        "truth_el_phi",
        "truth_el_charge",
        "truth_el_MCTC_isPrompt",
        "truth_el_q_over_pt",
    ]
    df = df[feat_cols].dropna().reset_index(drop=True)
    
    n_total = len(df)
    n_val = int(VAL_FRAC * n_total)
    df_train = df.sample(n=n_total - n_val, random_state=SEED)
    df_val = df.drop(df_train.index).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)

    def make_pairs(df, n_pairs):
        i = np.random.randint(0, len(df), size=n_pairs)
        j = np.random.randint(0, len(df), size=n_pairs)
        # ensure i != j
        j[i == j] = (j[i == j] + 1) % len(df)

        e1 = torch.tensor(df.iloc[i].values, dtype=torch.float32)
        e2 = torch.tensor(df.iloc[j].values, dtype=torch.float32)

        # Label: 1 = same charge (SC), 0 = opposite charge (OC)
        charge1 = df.iloc[i]["truth_el_charge"].values
        charge2 = df.iloc[j]["truth_el_charge"].values
        labels = torch.tensor((charge1 * charge2 > 0).astype(np.float32))

        return TensorDataset(e1, e2, labels)

    train_ds = make_pairs(df_train, len(df_train))
    val_ds = make_pairs(df_val, len(df_val))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        shuffle=True, #randomly shuffle the order of samples in each epoch
        pin_memory=True #copy tensors into page-locked memory
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        shuffle=False,
        pin_memory=True
    )

    input_dim = len(feat_cols)


    loss_cb = LossHistory()
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", #tells lightning to watch 'val_loss' metric
        mode="min",
        save_top_k=1, #keeps only the top k=1
        save_last=True, #saves last checkpoint from final training epoch
    )

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=DEVICE,
        logger=logger,
        callbacks=[loss_cb, ckpt_cb],
        log_every_n_steps=5,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Now compute predicted flip probability on the validation set
    # p(flip) = sigmoid( f_phi( phi(e1), phi(e2) ) )
    model.eval()
    with torch.no_grad():
        all_probs, all_y = [], []
        for e1, e2, y in val_loader:
            phi1, phi2 = model(e1), model(e2)
            f = f_phi(phi1, phi2)
            p = torch.sigmoid(f)
            all_probs.append(p.cpu().numpy())
            all_y.append(y.cpu().numpy())
        probs = np.concatenate(all_probs)
        ys = np.concatenate(all_y)
        print("Val mean p(flip)=", float(probs.mean()))

if __name__ == "__main__":
    main()