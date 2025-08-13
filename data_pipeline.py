import polars as pl

from typing import Tuple, List
from dataclasses import dataclass

from polars import DataFrame, LazyFrame

import logging

@dataclass
class DataConfig:
    data_path: str
    n_rows:int
    train_ratio: float
    feature_cols: List[str]
    target_col: str
    weight_col: str
    date_col:str

    @classmethod
    def from_dict(cls, data: dict) -> 'DataConfig':
        # Basic check for required keys before instantiation
        required = ['data_path', 'n_rows', 'train_ratio', 'feature_cols', 'target_col', 'weight_col', 'date_col']
        missing = [k for k in required if k not in data]
        if missing:
             # Maybe log this error instead of raising immediately?
             raise ValueError(f"Missing required keys for DataConfig: {missing}")
        return cls(**data)

class DataPipeline:
    def __init__(self, config: DataConfig, logger: logging.Logger):
        self.config = config
        self.logger: logging.Logger = logger

    def load_and_preprocess_data(self) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
        """Load and preprocess data, returning train and validation sets with normalization preventing leakage."""
        self.logger.info("Starting data loading and preprocessing.")
        # Data path is now expected to be the full path or glob pattern from config
        lf_initial = pl.scan_parquet(self.config.data_path).fill_null(3) # Consider default fill_null value

        # Determine actual feature columns if "auto" is used
        if isinstance(self.config.feature_cols, str) and self.config.feature_cols.lower() == "auto":
            self.logger.info("Feature columns set to 'auto', determining from data.")
            all_columns = lf_initial.columns
            actual_feature_cols = [
                col for col in all_columns
                if col not in [self.config.date_col, self.config.weight_col, self.config.target_col]
            ]
            self.logger.info(f"Determined {len(actual_feature_cols)} feature columns.")
        else:
            actual_feature_cols = self.config.feature_cols
            self.logger.info(f"Using {len(actual_feature_cols)} provided feature columns.")

        if not actual_feature_cols:
            self.logger.error("No feature columns determined or provided. Aborting.")
            raise ValueError("Feature columns list cannot be empty.")

        # Initial select, tail, and sort
        cols_for_query = [
            pl.col(self.config.date_col),
            pl.col(self.config.target_col),
            pl.col(self.config.weight_col),
            *[pl.col(f) for f in actual_feature_cols]
        ]

        # --- Debugging --- 
        self.logger.debug(f"Config - data_path: {self.config.data_path}")
        self.logger.debug(f"Config - n_rows: {self.config.n_rows}")
        self.logger.debug(f"Config - date_col: {self.config.date_col}")
        self.logger.debug(f"Config - target_col: {self.config.target_col}")
        self.logger.debug(f"Config - weight_col: {self.config.weight_col}")
        self.logger.debug(f"Config - feature_cols setting: {self.config.feature_cols}") # Show original setting
        self.logger.debug(f"Actual feature_cols being used ({len(actual_feature_cols)}): {actual_feature_cols[:10]}...") # Show first 10 actual
        self.logger.debug(f"Columns for query ({len(cols_for_query)}): {cols_for_query[:10]}...") # Show first 10 query cols
        try:
            initial_schema = lf_initial.schema
            self.logger.debug(f"Schema of lf_initial (before select/tail/sort): {initial_schema}")
        except Exception as e:
            self.logger.error(f"Failed to get schema of lf_initial: {e}")

        # --- End Debugging ---

        df_queried = (
            lf_initial.select(cols_for_query)
        .tail(self.config.n_rows)
            .sort(self.config.date_col)
            .collect()
        )
        self.logger.info(f"Collected initial DataFrame with shape: {df_queried.shape}")

        if df_queried.height == 0:
            self.logger.error("DataFrame is empty after initial query (select, tail, sort). Check data path and n_rows.")
            raise ValueError("Queried DataFrame is empty.")

        # Split data into training and validation sets (raw features)
        train_df_raw, val_df_raw = self._split_data_by_date(df_queried, actual_feature_cols)

        if train_df_raw.height == 0:
            self.logger.error("Training data subset is empty after split. Check date column and train_ratio.")
            raise ValueError("Training data subset is empty. Cannot calculate normalization stats.")

        # Calculate normalization statistics ONLY from the training set
        normalization_stats = self._calculate_normalization_stats(train_df_raw, actual_feature_cols)
        self.logger.info("Normalization statistics calculated from training data.")

        # Apply normalization to training set
        train_processed_df = self._apply_normalization(train_df_raw, normalization_stats, actual_feature_cols)
        self.logger.info(f"Training data normalized. Shape: {train_processed_df.shape}")

        # Apply normalization to validation set using stats from training set
        if val_df_raw.height > 0:
            val_processed_df = self._apply_normalization(val_df_raw, normalization_stats, actual_feature_cols)
            self.logger.info(f"Validation data normalized. Shape: {val_processed_df.shape}")
        else:
            self.logger.warning("Validation data subset is empty. Creating empty DataFrames for validation outputs.")
            val_df_schema = {f"{col}_normalized": pl.Float64 for col in actual_feature_cols}
            val_df = pl.DataFrame(schema=val_df_schema)
            val_target_schema = {f"{self.config.target_col}_normalized": pl.Float64}
            val_target = pl.DataFrame(schema=val_target_schema)
            weight_col_dtype = df_queried.schema.get(self.config.weight_col, pl.Float64) # Default dtype if not found
            val_weight_schema = {self.config.weight_col: weight_col_dtype}
            val_weight = pl.DataFrame(schema=val_weight_schema)


        # Extract final DataFrames for train
        train_df = train_processed_df.select([pl.col(f'{col}_normalized') for col in actual_feature_cols])
        train_target = train_processed_df.select(pl.col(f'{self.config.target_col}_normalized'))
        train_weight = train_processed_df.select(pl.col(self.config.weight_col))

        # Extract final DataFrames for validation (if val_processed_df exists and val_df_raw was not empty)
        if val_df_raw.height > 0: # Check ensures val_processed_df is defined
            val_df = val_processed_df.select([pl.col(f'{col}_normalized') for col in actual_feature_cols])
            val_target = val_processed_df.select(pl.col(f'{self.config.target_col}_normalized'))
            val_weight = val_processed_df.select(pl.col(self.config.weight_col))
        
        self.logger.info(f"Final train_df shape: {train_df.shape}, train_target shape: {train_target.shape}, train_weight shape: {train_weight.shape}")
        self.logger.info(f"Final val_df shape: {val_df.shape}, val_target shape: {val_target.shape}, val_weight shape: {val_weight.shape}")

        return train_df, train_target, train_weight, val_df, val_target, val_weight

    def _split_data_by_date(self, df: pl.DataFrame, feature_cols: List[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Split dataset into train and validation subsets based on date, keeping original features."""
        self.logger.info("Splitting data into train and validation sets by date.")
        unique_dates = df.get_column(self.config.date_col).unique().sort()
        
        if len(unique_dates) == 0:
            self.logger.error(f"No unique dates found in column: {self.config.date_col}. Cannot split data.")
            # Depending on requirements, either raise error or return empty DFs
            # For now, let's assume this is a critical error.
            raise ValueError(f"No unique dates found in column: {self.config.date_col} for splitting.")

        split_idx = int(len(unique_dates) * self.config.train_ratio)

        train_dates = unique_dates[:split_idx]
        val_dates = unique_dates[split_idx:]

        train_mask = df.get_column(self.config.date_col).is_in(train_dates)
        val_mask = df.get_column(self.config.date_col).is_in(val_dates)

        cols_to_select = [
            self.config.date_col,
            self.config.target_col,
            self.config.weight_col
        ] + feature_cols

        train_subset = df.filter(train_mask).select(cols_to_select)
        val_subset = df.filter(val_mask).select(cols_to_select)
        
        self.logger.info(f"Train subset shape: {train_subset.shape}, Val subset shape: {val_subset.shape}")
        if train_subset.height == 0 and self.config.train_ratio > 0: # Log if train is empty but was expected
            self.logger.warning("Training subset is empty after date split. Check date column distribution and train_ratio.")
        if val_subset.height == 0 and self.config.train_ratio < 1.0: # Log if val is empty but was expected
             self.logger.warning("Validation subset is empty after date split. Check date column distribution and train_ratio.")

        return train_subset, val_subset

    def _calculate_normalization_stats(self, df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
        """Calculate normalization statistics (quantiles, std) from the provided DataFrame."""
        self.logger.info(f"Calculating normalization statistics for {len(feature_cols)} features and target.")
        
        stats_expressions = []
        # For features
        for col in feature_cols:
            stats_expressions.extend([
                pl.col(col).quantile(0.05).alias(f"{col}_q05"),
                pl.col(col).quantile(0.95).alias(f"{col}_q95"),
                pl.col(col).std().alias(f"{col}_std")
            ])
        
        # For target column
        stats_expressions.extend([
            pl.col(self.config.target_col).quantile(0.05).alias(f"{self.config.target_col}_q05"),
            pl.col(self.config.target_col).quantile(0.95).alias(f"{self.config.target_col}_q95"),
            pl.col(self.config.target_col).std().alias(f"{self.config.target_col}_std")
        ])

        if df.height == 0:
            self.logger.error("Cannot calculate normalization stats from an empty DataFrame.")
            raise ValueError("Input DataFrame for calculating normalization stats is empty.")
            
        return df.lazy().select(stats_expressions).collect()

    def _apply_normalization(self, df: pl.DataFrame, stats: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
        """Apply normalization to DataFrame using pre-calculated stats."""
        self.logger.info(f"Applying normalization to a DataFrame with {df.height} rows.")
        
        if df.height == 0: # No need to normalize an empty frame, return it as is with expected schema
            self.logger.warning("Attempting to normalize an empty DataFrame. Returning empty frame with expected schema.")
            normalized_cols_schema = {f"{col}_normalized": pl.Float64 for col in feature_cols + [self.config.target_col]}
            # Need to determine dtypes for date_col and weight_col if df is truly empty
            # However, if df comes from _split_data_by_date, it would have schema.
            # This case is more for safety; typically df shouldn't be empty if stats were calculated.
            # For simplicity, we'll assume date/weight columns are just passed if present, or error if not.
            # A more robust way for empty df is to construct an empty DF with full target schema.
            # For now, let's select from the input 'df' which might already be empty but have schema.
            select_cols_for_empty = [self.config.date_col, self.config.weight_col] + \
                                    [f"{col}_normalized" for col in feature_cols + [self.config.target_col]]
            
            # Create empty columns with correct names and types if they don't exist
            empty_normalized_exprs = []
            for col_name in feature_cols + [self.config.target_col]:
                empty_normalized_exprs.append(pl.lit(None, dtype=pl.Float64).alias(f"{col_name}_normalized"))
            
            # Ensure date_col and weight_col are present or created as null if df is truly schemaless
            # This part is tricky if df is absolutely empty with no schema.
            # Assuming df has schema from split.
            if self.config.date_col not in df.columns:
                 df = df.with_columns(pl.lit(None, dtype=pl.Date).alias(self.config.date_col)) # Or appropriate date type
            if self.config.weight_col not in df.columns:
                 df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(self.config.weight_col))


            return df.with_columns(empty_normalized_exprs).select(select_cols_for_empty)


        normalized_exprs = []
        columns_to_normalize = feature_cols + [self.config.target_col]

        for col in columns_to_normalize:
            q05 = stats.get_column(f'{col}_q05')[0]
            q95 = stats.get_column(f'{col}_q95')[0]
            std_val = stats.get_column(f'{col}_std')[0]

            center = (q95 + q05) / 2.0
            
            scale_val_iqr = (q95 - q05) / 2.0
            
            if abs(scale_val_iqr) > 1e-10:
                scale = scale_val_iqr
            elif std_val is not None and abs(std_val) > 1e-10: # Check std_val is not None
                scale = std_val
            else:
                scale = 1.0
            
            # Final check for scale to prevent division by zero if all above are zero/None
            if abs(scale) < 1e-10:
                scale = 1.0
                self.logger.warning(f"Column '{col}' has near-zero scale (IQR and STD). Using scale=1.0 to avoid division by zero.")


            normalized_exprs.append(
                pl.when(pl.col(col).is_not_null() & (pl.col(col) > q95)) # Handle nulls before comparison
                .then(pl.lit(1.0, dtype=pl.Float64))
                .when(pl.col(col).is_not_null() & (pl.col(col) < q05)) # Handle nulls before comparison
                .then(pl.lit(-1.0, dtype=pl.Float64))
                .otherwise(
                    pl.when(pl.col(col).is_not_null()) # Apply scaling only to non-nulls
                    .then((pl.col(col) - center) / scale)
                    .otherwise(pl.lit(None, dtype=pl.Float64)) # Keep nulls as nulls
                 )
                .alias(f"{col}_normalized")
            )
        
        # Select original date and weight columns, and the new normalized columns
        return df.with_columns(normalized_exprs).select(
            pl.col(self.config.date_col),
            pl.col(self.config.weight_col),
            *[f"{col}_normalized" for col in columns_to_normalize]
        )

