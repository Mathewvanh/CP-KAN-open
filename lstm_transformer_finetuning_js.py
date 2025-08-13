import math
import itertools
import logging
import os
import sys
import time
from datetime import datetime
import gc
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# For older PyTorch versions (< 2.0) bfloat16 AMP usage:
from torch.amp import autocast, GradScaler

try:
    from data_pipeline_js_config import DataConfig
    from data_pipeline import DataPipeline
except ImportError:
    print("Warning: Could not import DataConfig or DataPipeline. Using placeholder classes.")
    # Define placeholder classes if imports fail
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
            # Simulate data loading for placeholder
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
            return (self.train_df, self.train_target, self.train_weight,
                    self.val_df, self.val_target, self.val_weight)


# --- Helper Functions ---

def count_parameters(module: nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted R² score."""
    numerator = np.sum(w * (y_true - y_pred)**2)
    denominator = np.sum(w * (y_true**2))
    if denominator < 1e-12:
        return 0.0
    numerator = max(0, numerator)
    return float(1.0 - numerator / denominator)

# --- Model Definitions ---

class LSTMModel(nn.Module):
    """
    Treat each feature as one time step:
    [batch_size, input_dim] -> [batch_size, input_dim, 1] -> embed -> LSTM -> final state -> output
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, embed_dim: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.embedding = nn.Linear(1, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, input_dim)
        x = x.unsqueeze(-1)           # (batch, input_dim, 1)
        x = self.embedding(x)         # (batch, input_dim, embed_dim)
        out, _ = self.lstm(x)         # (batch, input_dim, hidden_dim)
        out = out[:, -1, :]           # last time step => (batch, hidden_dim)
        out = self.fc(out)            # (batch, 1)
        return out


class GRUModel(nn.Module):
    """Same feature-as-sequence approach with GRU."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, embed_dim: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.embedding = nn.Linear(1, embed_dim)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)           # (batch, input_dim, 1)
        x = self.embedding(x)         # (batch, input_dim, embed_dim)
        out, _ = self.gru(x)          # (batch, input_dim, hidden_dim)
        out = out[:, -1, :]           # last time step => (batch, hidden_dim)
        out = self.fc(out)            # (batch, 1)
        return out


class TransformerModel(nn.Module):
    """
    Each feature is a 'token':
    [batch, input_dim] -> unsqueeze -> [batch, input_dim, 1] -> Linear(1->d_model) -> Transformer -> average pool -> fc
    """
    def __init__(self, input_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        self.input_fc = nn.Linear(1, d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout,
            batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.output_fc = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_fc.weight.data.uniform_(-initrange, initrange)
        self.input_fc.bias.data.zero_()
        self.output_fc.weight.data.uniform_(-initrange, initrange)
        self.output_fc.bias.data.zero_()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = src.unsqueeze(-1)               # [batch, input_dim, 1]
        src = self.input_fc(src)             # [batch, input_dim, d_model]
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)          # average pool over the sequence dim
        output = self.output_fc(output)      # [batch, 1]
        return output


# --- Grid Search ---

def run_grid_search(
    target_model_type: str = 'all',
    results_filename="results_js/recurrent_transformer_grid_search.csv"
):
    # Setup logging
    log_file = f'recurrent_transformer_grid_search_{target_model_type}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    logger = logging.getLogger("GridSearch")
    logger.info("Starting Recurrent/Transformer Grid Search...")

    # --- Config ---
    data_cfg = DataConfig(
        data_path="/Users/darklight/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/**/*.parquet",
        n_rows=200000,
        train_ratio=0.7,
        feature_cols=[f'feature_{i:02d}' for i in range(79)],  # example
        target_col="responder_6",
        weight_col="weight",
        date_col="date_id"
    )
    num_epochs = 50
    patience = 5
    weight_decay = 1e-5

    # Hyperparameter grids
    common_grid = {
        'lr': [1e-3, 5e-4, 1e-4],
        'batch_size': [512, 1024],
        'dropout': [0.1, 0.2],
    }
    recurrent_grid = {
        'hidden_dim': [32, 64],
        'num_layers': [1, 2],
    }
    transformer_grid = {
        'd_model': [32, 64],
        'nhead': [4, 8],
        'd_hid': [64, 128],
        'nlayers': [1, 2],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        # Disable certain advanced attention kernels that can cause problems on some GPUs
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # --- Data ---
    logger.info("Loading data...")
    pipeline = DataPipeline(data_cfg, logger)
    try:
        train_df, train_target, train_weight, val_df, val_target, val_weight = pipeline.load_and_preprocess_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        if isinstance(pipeline, DataPipeline):
            train_df, train_target, train_weight, val_df, val_target, val_weight = pipeline.load_and_preprocess_data()
        else:
            return

    # Convert DataFrame to Tensors
    x_train = torch.tensor(train_df.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(train_target.to_numpy(), dtype=torch.float32).squeeze(-1)
    w_train = torch.tensor(train_weight.to_numpy(), dtype=torch.float32).squeeze(-1)

    x_val = torch.tensor(val_df.to_numpy(), dtype=torch.float32)
    y_val = torch.tensor(val_target.to_numpy(), dtype=torch.float32).squeeze(-1)
    w_val = torch.tensor(val_weight.to_numpy(), dtype=torch.float32).squeeze(-1)

    input_dim = x_train.shape[1]
    logger.info(f"Data loaded: Train shape={x_train.shape}, Val shape={x_val.shape}, Input dim={input_dim}")

    # --- Results ---
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    results_cols = [
        'model_type', 'hidden_dim', 'num_layers', 'd_model', 'nhead',
        'd_hid', 'nlayers', 'dropout', 'lr', 'batch_size', 'param_count',
        'best_epoch', 'best_val_r2', 'final_train_r2', 'run_timestamp'
    ]
    if os.path.exists(results_filename):
        results_df = pd.read_csv(results_filename)
        logger.info(f"Loaded existing results from {results_filename}")
    else:
        results_df = pd.DataFrame(columns=results_cols)
        logger.info(f"Created new results file: {results_filename}")

    # --- Model Lists ---
    all_models_to_search = {
        'LSTM': (LSTMModel, {**common_grid, **recurrent_grid}),
        'GRU': (GRUModel, {**common_grid, **recurrent_grid}),
        'Transformer': (TransformerModel, {**common_grid, **transformer_grid})
    }

    if target_model_type == 'all':
        models_to_search = all_models_to_search
        logger.info("Running grid search for ALL model types (LSTM, GRU, Transformer).")
    elif target_model_type in all_models_to_search:
        models_to_search = {target_model_type: all_models_to_search[target_model_type]}
        logger.info(f"Running grid search ONLY for model type: {target_model_type}")
    else:
        logger.error("Invalid model_type specified.")
        return

    # Count total combos
    total_combinations = 0
    for model_name, (_, grid) in models_to_search.items():
        keys, values = zip(*grid.items())
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
        total_combinations += len(combos)
    logger.info(f"Total hyperparameter combinations to test: {total_combinations}")

    # For AMP on older PyTorch
    scaler = GradScaler()

    run_counter = 0
    for model_name, (model_class, grid) in models_to_search.items():
        keys, values = zip(*grid.items())
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for i, params in enumerate(combos):
            run_counter += 1
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"\n--- Run {run_counter}/{total_combinations}: {model_name} | Params: {params} ---")

            # For Transformers, ensure d_model divisible by nhead
            if model_name == 'Transformer':
                if params['d_model'] % params['nhead'] != 0:
                    logger.warning(f"Skipping combo: d_model ({params['d_model']}) must be divisible by nhead ({params['nhead']}).")
                    continue

            # Instantiate
            model_specific_params = {}
            if model_name in ['LSTM', 'GRU']:
                model_specific_params = {k: params[k] for k in ['hidden_dim', 'num_layers'] if k in params}
            elif model_name == 'Transformer':
                model_specific_params = {k: params[k] for k in ['d_model', 'nhead', 'd_hid', 'nlayers']}

            try:
                model = model_class(
                    input_dim=input_dim,
                    dropout=params['dropout'],
                    **model_specific_params
                ).to(device)
            except Exception as e:
                logger.error(f"Error instantiating {model_name} with params {params}: {e}")
                continue

            param_count = count_parameters(model)
            logger.info(f"Model: {model_name}, Parameters: {param_count}")

            # DataLoaders
            batch_size = params['batch_size']
            train_dataset = TensorDataset(x_train, y_train, w_train)
            val_dataset = TensorDataset(x_val, y_val, w_val)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                pin_memory=True if device.type == 'cuda' else False
            )

            # Move val data to device
            x_val_dev = x_val.to(device)
            y_val_dev = y_val.to(device)
            w_val_dev = w_val.to(device)

            # Also move full train data for computing train R²
            x_train_dev = x_train.to(device)
            y_train_dev = y_train.to(device)
            w_train_dev = w_train.to(device)

            # Optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=weight_decay)

            # Training
            best_val_r2 = -float('inf')
            best_epoch = -1
            patience_counter = 0
            epoch_train_r2 = -float('inf')
            train_start_time = time.time()

            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                model.train()
                train_loss_num = 0.0
                train_loss_den = 0.0

                for x_batch, y_batch, w_batch in train_loader:
                    x_batch, y_batch, w_batch = x_batch.to(device), y_batch.to(device), w_batch.to(device)
                    optimizer.zero_grad()

                    # Mixed precision forward
                    with autocast(dtype=torch.bfloat16, device_type="cuda"):
                        y_pred = model(x_batch).squeeze(-1)
                        numerator = torch.sum(w_batch * (y_batch - y_pred)**2)
                        denominator = torch.sum(w_batch)
                        loss = numerator / (denominator + 1e-12)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss_num += numerator.detach().item()
                    train_loss_den += denominator.item()

                avg_train_loss = train_loss_num / (train_loss_den + 1e-12)

                # Evaluation
                model.eval()
                with torch.no_grad():
                    # Validation predictions
                    with autocast(dtype=torch.bfloat16, device_type="cuda"):
                        y_pred_val = model(x_val_dev).squeeze(-1)
                    # Weighted R² on validation set
                    val_r2 = weighted_r2(
                        y_val_dev.cpu().numpy(),
                        y_pred_val.to(torch.float32).cpu().numpy(),
                        w_val_dev.cpu().numpy()
                    )

                    # Training predictions
                    with autocast(dtype=torch.bfloat16, device_type="cuda"):
                        y_pred_train = model(x_train_dev).squeeze(-1)
                    # Weighted R² on training set
                    epoch_train_r2 = weighted_r2(
                        y_train_dev.cpu().numpy(),
                        y_pred_train.to(torch.float32).cpu().numpy(),  # <-- the fix
                        w_train_dev.cpu().numpy()
                    )

                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} "
                    f"| Train Loss: {avg_train_loss:.4f} "
                    f"| Train R²: {epoch_train_r2:.4f} "
                    f"| Val R²: {val_r2:.4f} "
                    f"| Time: {epoch_time:.2f}s"
                )

                # Early Stopping
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch+1}. "
                            f"Best Val R²: {best_val_r2:.4f} at epoch {best_epoch}"
                        )
                        break

            train_time = time.time() - train_start_time
            logger.info(f"Training finished. Total time: {train_time:.2f}s")

            # Record results
            result_row = {
                'model_type': model_name,
                'param_count': param_count,
                'best_epoch': best_epoch,
                'best_val_r2': best_val_r2,
                'final_train_r2': epoch_train_r2,
                'run_timestamp': run_timestamp
            }
            result_row.update(params)
            for col in ['hidden_dim', 'num_layers', 'd_model', 'nhead', 'd_hid', 'nlayers']:
                if col not in result_row:
                    result_row[col] = None

            new_row_df = pd.DataFrame([result_row])
            new_row_df = new_row_df.reindex(columns=results_df.columns)

            for col in results_df.columns:
                if col in new_row_df.columns:
                    try:
                        new_row_df[col] = new_row_df[col].astype(results_df[col].dtype, errors='ignore')
                    except Exception:
                        pass

            results_df = pd.concat([results_df, new_row_df], ignore_index=True)

            try:
                results_df.to_csv(results_filename, index=False)
            except IOError as e:
                logger.error(f"Could not save results to {results_filename}: {e}")

            # Cleanup
            del model, optimizer, train_loader, train_dataset, val_dataset
            del x_batch, y_batch, w_batch, y_pred, y_pred_val, y_pred_train
            del x_val_dev, y_val_dev, w_val_dev, x_train_dev, y_train_dev, w_train_dev
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU Memory Cleared.")

    logger.info("Grid Search Completed.")
    logger.info(f"Results saved to {results_filename}")
    logger.info(f"Log saved to {log_file}")


# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hyperparameter grid search for sequence models on data."
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='all',
        choices=['LSTM', 'GRU', 'Transformer', 'all'],
        help="Model type for grid search."
    )
    args = parser.parse_args()

    os.makedirs("results_js", exist_ok=True)

    run_grid_search(target_model_type=args.model_type)