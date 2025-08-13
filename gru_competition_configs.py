import logging
import os
import sys
import time
import gc
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# For older PyTorch versions that don't support device_type in autocast:
from torch.amp import autocast, GradScaler

# We'll add a progress bar for clearer logging:
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bars...")
    import subprocess
    subprocess.run(["pip", "install", "tqdm"], check=True)
    from tqdm import tqdm

# ---------------- If your real code can import these, do so. Otherwise placeholders:
try:
    from data_pipeline_js_config import DataConfig
    from data_pipeline import DataPipeline
except ImportError:
    print("Warning: Could not import DataConfig or DataPipeline. Using placeholder classes.")
    class DataConfig:
        def __init__(self, data_path, n_rows, train_ratio, feature_cols, target_col, weight_col, date_col):
            self.data_path = data_path
            self.n_rows = n_rows
            self.train_ratio = train_ratio
            self.feature_cols = feature_cols
            self.target_col = target_col
            self.weight_col = weight_col
            self.date_col = date_col

    class DataPipeline:
        def __init__(self, config, logger):
            self.config = config
            self.logger = logger
            self.logger.info("Using placeholder DataPipeline.")
            n_features = len(config.feature_cols)
            n_train = int(config.n_rows * config.train_ratio)
            n_val = config.n_rows - n_train

            self.train_df = pd.DataFrame(np.random.randn(n_train, n_features), columns=config.feature_cols)
            self.train_target = pd.DataFrame(np.random.randn(n_train, 1), columns=[config.target_col])
            self.train_weight = pd.DataFrame(np.abs(np.random.rand(n_train, 1)), columns=[config.weight_col])
            self.val_df = pd.DataFrame(np.random.randn(n_val, n_features), columns=config.feature_cols)
            self.val_target = pd.DataFrame(np.random.randn(n_val, 1), columns=[config.target_col])
            self.val_weight = pd.DataFrame(np.abs(np.random.rand(n_val, 1)), columns=[config.weight_col])

        def load_and_preprocess_data(self):
            self.logger.info("Placeholder: Loading and preprocessing dummy data...")
            return (
                self.train_df, self.train_target, self.train_weight,
                self.val_df, self.val_target, self.val_weight
            )


# ---------------- Helper Functions ----------------

def weighted_r2(y_true, y_pred, w):
    """Compute weighted R² score."""
    numerator = np.sum(w * (y_true - y_pred)**2)
    denominator = np.sum(w * (y_true**2))
    if denominator < 1e-12:
        return 0.0
    numerator = max(0, numerator)
    return 1.0 - numerator / denominator

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_r2_in_batches(model, dataloader, device, autocast_dtype=torch.bfloat16):
    """
    Compute Weighted R² over an entire dataset in mini-batches to avoid OOM.
    Assumes dataloader yields (x_batch, y_batch, w_batch).
    """
    model.eval()
    total_num = 0.0
    total_den = 0.0

    with torch.no_grad():
        for x_batch, y_batch, w_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)

            with autocast('cuda',dtype=autocast_dtype) if device.type == 'cuda' else torch.enable_grad():
                preds = model(x_batch).squeeze(-1)  # [batch_size]
            
            # Move back to CPU for the R² calculations
            preds_cpu = preds.to(torch.float32).cpu().numpy()
            y_cpu = y_batch.cpu().numpy()
            w_cpu = w_batch.cpu().numpy()

            total_num += np.sum(w_cpu * (y_cpu - preds_cpu)**2)
            total_den += np.sum(w_cpu * (y_cpu**2))

    if total_den < 1e-12:
        return 0.0
    total_num = max(0, total_num)
    return 1.0 - (total_num / total_den)

# ---------------- Model Definition ----------------

class CompetitionGRUModel(nn.Module):
    """
    A flexible GRU model that:
    1) Takes input [batch_size, input_dim].
    2) Interprets each feature as a time-step => [batch_size, input_dim, 1].
    3) Builds multiple GRU layers & dropout.
    4) Final hidden state -> linear stack -> [batch_size, 1].
    """
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list,         # e.g. [500] or [250,150,150]
        dropout_rates: list,        # e.g. [0.3,0.0,0.0], one per GRU layer
        hidden_sizes_linear: list,  # e.g. [500,300]
        dropout_rates_linear: list, # e.g. [0.2,0.1]
        embed_as_sequence: bool = True
    ):
        super().__init__()

        self.embed_as_sequence = embed_as_sequence
        if embed_as_sequence:
            # We'll treat each feature as one "time step"
            self.input_dim = input_dim
            self.input_embed_dim = 1
        else:
            # Single time step => [batch_size, 1, input_dim]
            self.input_dim = 1
            self.input_embed_dim = input_dim

        self.gru_layers = nn.ModuleList()
        self.dropouts_gru = nn.ModuleList()

        prev_size = self.input_embed_dim
        for idx, hsize in enumerate(hidden_sizes):
            layer = nn.GRU(
                input_size=prev_size,
                hidden_size=hsize,
                batch_first=True
            )
            self.gru_layers.append(layer)
            self.dropouts_gru.append(nn.Dropout(dropout_rates[idx]))
            prev_size = hsize

        # Build the linear "head"
        linear_layers = []
        in_features = prev_size
        for l_idx, lin_size in enumerate(hidden_sizes_linear):
            linear_layers.append(nn.Linear(in_features, lin_size))
            linear_layers.append(nn.ReLU())
            if dropout_rates_linear[l_idx] > 0:
                linear_layers.append(nn.Dropout(dropout_rates_linear[l_idx]))
            in_features = lin_size
        # Final output layer
        linear_layers.append(nn.Linear(in_features, 1))

        self.linear_stack = nn.Sequential(*linear_layers)

    def forward(self, x):
        # x: [batch_size, input_dim]
        if self.embed_as_sequence:
            x = x.unsqueeze(-1)  # => [batch_size, input_dim, 1]
        else:
            x = x.unsqueeze(1)   # => [batch_size, 1, input_dim]

        out = x
        for layer, dropout in zip(self.gru_layers, self.dropouts_gru):
            out, h = layer(out)  # shape => [batch_size, seq_len, hidden_size]
            out = dropout(out)

        # final hidden state => out[:, -1, :]
        out = out[:, -1, :]
        out = self.linear_stack(out)  # => [batch_size, 1]
        return out

# ---------------- Competition Configs ----------------

COMPETITION_GRU_CONFIGS = [
    {
        'name': 'gru_2.0_700',
        'model_type': 'gru',
        'hidden_sizes': [500],
        'dropout_rates': [0.3, 0.0, 0.0],  # We'll ignore extra zeros beyond the first layer
        'hidden_sizes_linear': [500, 300],
        'dropout_rates_linear': [0.2, 0.1],
        'lr': 0.0005,
        'batch_size': 512,  # 1 is too slow; let's do 512 now
        'epochs': 8,
        'early_stopping_patience': 1,
        'early_stopping': False,
        'lr_patience': 10,
        'lr_factor': 0.5,
        'lr_refit': 0.0003,
        'random_seed': 0
    },
    {
        'name': 'gru_2.1_700',
        'model_type': 'gru',
        'hidden_sizes': [500],
        'dropout_rates': [0.3, 0.0, 0.0],
        'hidden_sizes_linear': [500, 300],
        'dropout_rates_linear': [0.2, 0.1],
        'lr': 0.0005,
        'batch_size': 512,
        'epochs': 8,
        'early_stopping_patience': 1,
        'early_stopping': False,
        'lr_patience': 10,
        'lr_factor': 0.5,
        'lr_refit': 0.0003,
        'random_seed': 1
    },
    {
        'name': 'gru_2.2_700',
        'model_type': 'gru',
        'hidden_sizes': [500],
        'dropout_rates': [0.3, 0.0, 0.0],
        'hidden_sizes_linear': [500, 300],
        'dropout_rates_linear': [0.2, 0.1],
        'lr': 0.0005,
        'batch_size': 512,
        'epochs': 8,
        'early_stopping_patience': 1,
        'early_stopping': False,
        'lr_patience': 10,
        'lr_factor': 0.5,
        'lr_refit': 0.0003,
        'random_seed': 2
    },
    {
        'name': 'gru_3.0_700',
        'model_type': 'gru',
        'hidden_sizes': [250, 150, 150],
        'dropout_rates': [0.0, 0.0, 0.0],
        'hidden_sizes_linear': [],
        'dropout_rates_linear': [],
        'lr': 0.0005,
        'batch_size': 512,
        'epochs': 8,
        'early_stopping_patience': 1,
        'early_stopping': False,
        'lr_patience': 10,
        'lr_factor': 0.5,
        'lr_refit': 0.0003,
        'random_seed': 0
    },
    {
        'name': 'gru_3.1_700',
        'model_type': 'gru',
        'hidden_sizes': [250, 150, 150],
        'dropout_rates': [0.0, 0.0, 0.0],
        'hidden_sizes_linear': [],
        'dropout_rates_linear': [],
        'lr': 0.0005,
        'batch_size': 512,
        'epochs': 8,
        'early_stopping_patience': 1,
        'early_stopping': False,
        'lr_patience': 10,
        'lr_factor': 0.5,
        'lr_refit': 0.0003,
        'random_seed': 1
    },
    {
        'name': 'gru_3.2_700',
        'model_type': 'gru',
        'hidden_sizes': [250, 150, 150],
        'dropout_rates': [0.0, 0.0, 0.0],
        'hidden_sizes_linear': [],
        'dropout_rates_linear': [],
        'lr': 0.0005,
        'batch_size': 512,
        'epochs': 8,
        'early_stopping_patience': 1,
        'early_stopping': False,
        'lr_patience': 10,
        'lr_factor': 0.5,
        'lr_refit': 0.0003,
        'random_seed': 2
    },
]

def main():
    parser = argparse.ArgumentParser(description="Run competition-style GRU configs.")
    parser.add_argument("--results_csv", type=str, default="results_js/gru_competition_experiments.csv",
                        help="Where to save results.")
    args = parser.parse_args()

    # Setup logging
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/gru_competition.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, mode='a')]
    )
    logger = logging.getLogger("GRUCompetition")

    # Some basic config for data: 79 features
    data_cfg = DataConfig(
        data_path="~/kaggle-normal/JS_finetune/train.parquet/",
        n_rows=200000,   # Adjust as needed
        train_ratio=0.7,
        feature_cols=[f'feature_{i:02d}' for i in range(79)],  # 79 features
        target_col="responder_6",
        weight_col="weight",
        date_col="date_id"
    )

    logger.info("Loading data pipeline...")
    pipeline = DataPipeline(data_cfg, logger)
    train_df, train_target, train_weight, val_df, val_target, val_weight = pipeline.load_and_preprocess_data()

    # Convert to torch Tensors (stay on CPU for now)
    x_train = torch.tensor(train_df.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(train_target.to_numpy(), dtype=torch.float32).squeeze(-1)
    w_train = torch.tensor(train_weight.to_numpy(), dtype=torch.float32).squeeze(-1)

    x_val = torch.tensor(val_df.to_numpy(), dtype=torch.float32)
    y_val = torch.tensor(val_target.to_numpy(), dtype=torch.float32).squeeze(-1)
    w_val = torch.tensor(val_weight.to_numpy(), dtype=torch.float32).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Create or load results CSV
    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    if os.path.exists(args.results_csv):
        results_df = pd.read_csv(args.results_csv)
        logger.info(f"Loaded existing results from {args.results_csv}")
    else:
        results_df = pd.DataFrame(columns=[
            "experiment_name", "model_type", "hidden_sizes", "dropout_rates",
            "hidden_sizes_linear", "dropout_rates_linear", "lr",
            "batch_size", "epochs", "seed", "param_count",
            "final_val_r2", "final_train_r2"
        ])
        logger.info(f"Will create new results file at {args.results_csv}")

    # AMP
    scaler = GradScaler()

    # We'll create separate DataLoaders for training (for training loop) and for R² eval:
    # Because we want to do a mini-batch pass over the entire train dataset to compute R²,
    # so we can do so in the same memory-friendly approach.
    for config in COMPETITION_GRU_CONFIGS:
        logger.info(f"\n=== Starting experiment: {config['name']} ===")

        exp_name = config['name']
        seed = config['random_seed']
        set_seed(seed)

        # Create Datasets (CPU Tensors)
        train_dataset = TensorDataset(x_train, y_train, w_train)
        val_dataset = TensorDataset(x_val, y_val, w_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            pin_memory=(device.type == 'cuda')
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            pin_memory=(device.type == 'cuda')
        )

        # Build the GRU model
        n_gru_layers = len(config['hidden_sizes'])
        dropout_rates = config['dropout_rates'][:n_gru_layers]

        n_lin_layers = len(config['hidden_sizes_linear'])
        dropout_rates_linear = config['dropout_rates_linear'][:n_lin_layers]

        model = CompetitionGRUModel(
            input_dim=x_train.shape[1],
            hidden_sizes=config['hidden_sizes'],
            dropout_rates=dropout_rates,
            hidden_sizes_linear=config['hidden_sizes_linear'],
            dropout_rates_linear=dropout_rates_linear,
            embed_as_sequence=True
        ).to(device)

        param_count = count_parameters(model)
        logger.info(f"Model param count: {param_count}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
        epochs = config['epochs']
        best_val_r2 = -float('inf')
        best_epoch = -1
        patience_counter = 0
        early_stopping_patience = config['early_stopping_patience']
        do_early_stopping = config['early_stopping']

        for epoch in range(epochs):
            model.train()
            running_loss_num = 0.0
            running_loss_den = 0.0
            t0 = time.time()

            # Training loop
            for x_batch, y_batch, w_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                w_batch = w_batch.to(device)
                optimizer.zero_grad()

                with autocast('cuda',dtype=torch.bfloat16) if device.type == 'cuda' else torch.enable_grad():
                    y_pred = model(x_batch).squeeze(-1)
                    numerator = torch.sum(w_batch * (y_batch - y_pred)**2)
                    denominator = torch.sum(w_batch)
                    loss = numerator / (denominator + 1e-12)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss_num += numerator.detach().item()
                running_loss_den += denominator.item()

            avg_train_loss = running_loss_num / (running_loss_den + 1e-12)

            # Evaluate Weighted R² on train and val sets in mini-batches
            train_r2 = evaluate_r2_in_batches(model, train_loader, device, autocast_dtype=torch.bfloat16)
            val_r2 = evaluate_r2_in_batches(model, val_loader, device, autocast_dtype=torch.bfloat16)

            epoch_time = time.time() - t0
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"TrainLoss={avg_train_loss:.4f}, TrainR2={train_r2:.4f}, ValR2={val_r2:.4f}, Time={epoch_time:.2f}s"
            )

            # Early Stopping
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if do_early_stopping and (patience_counter >= early_stopping_patience):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch}, ValR2={best_val_r2:.4f}")
                    break

        final_train_r2 = train_r2
        final_val_r2 = best_val_r2

        logger.info(f"[{exp_name}] Finished. Best epoch={best_epoch}, ValR2={best_val_r2:.4f}, FinalTrainR2={final_train_r2:.4f}")

        # Record results
        row_dict = {
            "experiment_name": exp_name,
            "model_type": "gru",
            "hidden_sizes": config['hidden_sizes'],
            "dropout_rates": config['dropout_rates'],
            "hidden_sizes_linear": config['hidden_sizes_linear'],
            "dropout_rates_linear": config['dropout_rates_linear'],
            "lr": config['lr'],
            "batch_size": config['batch_size'],
            "epochs": config['epochs'],
            "seed": seed,
            "param_count": param_count,
            "final_val_r2": final_val_r2,
            "final_train_r2": final_train_r2
        }
        results_df = pd.concat([results_df, pd.DataFrame([row_dict])], ignore_index=True)

        # Cleanup
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    # Save results
    results_df.to_csv(args.results_csv, index=False)
    logger.info(f"All experiments complete. Results saved to {args.results_csv}")


if __name__ == "__main__":
    main()