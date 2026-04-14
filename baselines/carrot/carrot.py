"""
CARROT (Cost-Aware Router with Regressor-based Optimization Techniques)

This module provides LLM routing with quality and cost prediction using
K-Nearest Neighbors and Linear Regression baselines.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import joblib


# =============================================================================
# Data Utilities
# =============================================================================

def load_and_align_data(data_dir: str,
                        encoder_model: str,
                        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load CSV files, align on common prompts, and compute embeddings.

    Args:
        data_dir: Directory containing CSV files with scored data
        encoder_model: Sentence transformer model name for embeddings
        verbose: Whether to print progress messages

    Returns:
        Tuple of (embeddings, quality_scores, token_counts, model_names, prompts):
            - embeddings: np.ndarray of shape (n_samples, embedding_dim)
            - quality_scores: np.ndarray of shape (n_samples, n_models)
            - token_counts: np.ndarray of shape (n_samples, n_models)
            - model_names: List of model names
            - prompts: List of prompts
    """
    if verbose:
        print("\n=== Loading Data ===")

    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*_scored.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    if verbose:
        print(f"Found {len(csv_files)} CSV files:")
        for f in csv_files:
            print(f"  - {f.name}")

    # Load all dataframes
    dfs = {}
    for csv_file in csv_files:
        model_name = csv_file.stem.replace("_scored", "").replace("_eval", "")
        df = pd.read_csv(csv_file)
        if verbose:
            print(f"\nLoaded {model_name}: {len(df)} rows")

        # Check required columns
        required_cols = ['prompt', 'judge_score', 'output_tokens']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {csv_file.name}: {missing}")

        dfs[model_name] = df

    # Find common prompts across all models
    if verbose:
        print("\n=== Aligning Data ===")

    common_prompts = None
    for model_name, df in dfs.items():
        prompts_set = set(df['prompt'].values)
        if common_prompts is None:
            common_prompts = prompts_set
        else:
            common_prompts = common_prompts.intersection(prompts_set)

    if verbose:
        print(f"Common prompts across all models (before NaN filtering): {len(common_prompts)}")

    if len(common_prompts) == 0:
        raise ValueError("No common prompts found across all CSV files!")

    # Create aligned dataframes and filter NaN
    aligned_dfs = {}
    for model_name, df in dfs.items():
        df_filtered = df[df['prompt'].isin(common_prompts)].copy()

        # Remove rows with NaN in judge_score or output_tokens
        before_count = len(df_filtered)
        df_filtered = df_filtered.dropna(subset=['judge_score', 'output_tokens'])
        after_nan = len(df_filtered)

        # Remove duplicate prompts (keep first occurrence)
        df_filtered = df_filtered.drop_duplicates(subset=['prompt'], keep='first')
        after_dedup = len(df_filtered)

        # Sort by prompt
        df_filtered = df_filtered.sort_values('prompt').reset_index(drop=True)

        if verbose and before_count != after_dedup:
            print(f"  {model_name}: Removed {before_count - after_nan} NaN rows, {after_nan - after_dedup} duplicates")

        aligned_dfs[model_name] = df_filtered

    # Find prompts that have valid data across all models
    valid_prompts = None
    for model_name, df in aligned_dfs.items():
        prompts_set = set(df['prompt'].values)
        if valid_prompts is None:
            valid_prompts = prompts_set
        else:
            valid_prompts = valid_prompts.intersection(prompts_set)

    if verbose:
        print(f"Valid prompts across all models (after NaN filtering): {len(valid_prompts)}")

    # Re-filter to only valid prompts
    for model_name in aligned_dfs.keys():
        df = aligned_dfs[model_name]
        aligned_dfs[model_name] = df[df['prompt'].isin(valid_prompts)].sort_values('prompt').reset_index(drop=True)
        if verbose:
            print(f"  {model_name}: {len(aligned_dfs[model_name])} final samples")

    # Verify alignment
    prompts_list = aligned_dfs[list(aligned_dfs.keys())[0]]['prompt'].values
    for model_name, df in aligned_dfs.items():
        if not np.array_equal(df['prompt'].values, prompts_list):
            raise ValueError(f"Prompt alignment failed for {model_name}")

    if verbose:
        print(f"\n✅ All models aligned on {len(prompts_list)} prompts")

    # Extract quality scores and token counts
    model_names = list(aligned_dfs.keys())
    quality_scores = np.column_stack([
        aligned_dfs[name]['judge_score'].values for name in model_names
    ])
    token_counts = np.column_stack([
        aligned_dfs[name]['output_tokens'].values for name in model_names
    ])

    if verbose:
        print(f"\nQuality scores shape: {quality_scores.shape}")
        print(f"Token counts shape: {token_counts.shape}")

    # Compute embeddings
    if verbose:
        print(f"\n=== Computing Embeddings ===")
        print(f"Using model: {encoder_model}")

    encoder = SentenceTransformer(encoder_model)
    embeddings = encoder.encode(prompts_list, show_progress_bar=verbose, convert_to_numpy=True)

    if verbose:
        print(f"Embeddings shape: {embeddings.shape}")

    return embeddings, quality_scores, token_counts, model_names, prompts_list.tolist()


def filter_predictions_to_models(predictions: np.ndarray,
                                 trained_model_names: List[str],
                                 eval_model_names: List[str]) -> np.ndarray:
    """
    Filter predictions to match evaluation model subset.

    Args:
        predictions: Prediction array of shape (n_samples, n_trained_models)
        trained_model_names: List of model names from training
        eval_model_names: List of model names for evaluation

    Returns:
        Filtered predictions of shape (n_samples, n_eval_models)
    """
    eval_model_indices = []
    for eval_model in eval_model_names:
        # Remove _eval suffix to match training names
        base_name = eval_model.replace('_eval', '')
        if base_name in trained_model_names:
            eval_model_indices.append(trained_model_names.index(base_name))
        else:
            raise ValueError(f"Eval model '{base_name}' not found in training models: {trained_model_names}")

    return predictions[:, eval_model_indices]


# =============================================================================
# CARROT Baselines
# =============================================================================

class CarrotKNNBaseline:
    """CARROT baseline using K-Nearest Neighbors regression."""

    def __init__(self, n_neighbors_score: int = 256, n_neighbors_count: int = 256,
                 metric: str = "cosine", load_dir: str = None):
        """
        Initialize CARROT-KNN baseline.

        Args:
            n_neighbors_score: Number of neighbors for quality prediction
            n_neighbors_count: Number of neighbors for token count prediction
            metric: Distance metric (default: cosine)
            load_dir: Directory to load pre-trained models from (optional)
        """
        self.n_neighbors_score = n_neighbors_score
        self.n_neighbors_count = n_neighbors_count
        self.metric = metric

        if load_dir:
            self.load(load_dir)
        else:
            self.knn_score = None
            self.knn_count = None

    def fit(self, embedding_train: np.ndarray, quality_train: np.ndarray,
            token_count_train: np.ndarray, save_dir: str = None):
        """
        Fit KNN models on training data.

        Args:
            embedding_train: Training embeddings (n_samples, embedding_dim)
            quality_train: Training quality scores (n_samples, n_models)
            token_count_train: Training token counts (n_samples, n_models)
            save_dir: Directory to save trained models (optional)
        """
        print("🚀 Training CARROT-KNN...")

        # Train KNN for quality prediction
        self.knn_score = KNeighborsRegressor(
            n_neighbors=self.n_neighbors_score,
            metric=self.metric
        )
        self.knn_score.fit(embedding_train, quality_train)

        # Train KNN for token count prediction
        self.knn_count = KNeighborsRegressor(
            n_neighbors=self.n_neighbors_count,
            metric=self.metric
        )
        self.knn_count.fit(embedding_train, token_count_train)

        print("✅ CARROT-KNN training complete")

        if save_dir:
            self.save(save_dir)

    def predict(self, embedding_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict quality scores and token counts.

        Args:
            embedding_test: Test embeddings (n_samples, embedding_dim)

        Returns:
            Tuple of (quality_pred, token_count_pred)
        """
        quality_pred = self.knn_score.predict(embedding_test)
        token_count_pred = self.knn_count.predict(embedding_test)
        return quality_pred, token_count_pred

    def save(self, save_dir: str):
        """Save trained models to directory."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.knn_score, f"{save_dir}/knn_score.joblib")
        joblib.dump(self.knn_count, f"{save_dir}/knn_count.joblib")
        print(f"💾 Saved CARROT-KNN models to {save_dir}")

    def load(self, load_dir: str):
        """Load trained models from directory."""
        self.knn_score = joblib.load(f"{load_dir}/knn_score.joblib")
        self.knn_count = joblib.load(f"{load_dir}/knn_count.joblib")


class CarrotLinearBaseline:
    """CARROT baseline using Linear Regression."""

    def __init__(self, fit_intercept: bool = True, load_dir: str = None):
        """
        Initialize CARROT-Linear baseline.

        Args:
            fit_intercept: Whether to fit intercept (default: True)
            load_dir: Directory to load pre-trained models from (optional)
        """
        self.fit_intercept = fit_intercept

        if load_dir:
            self.load(load_dir)
        else:
            self.linear_score = None
            self.linear_count = None

    def fit(self, embedding_train: np.ndarray, quality_train: np.ndarray,
            token_count_train: np.ndarray, save_dir: str = None):
        """
        Fit Linear Regression models on training data.

        Args:
            embedding_train: Training embeddings (n_samples, embedding_dim)
            quality_train: Training quality scores (n_samples, n_models)
            token_count_train: Training token counts (n_samples, n_models)
            save_dir: Directory to save trained models (optional)
        """
        print("🚀 Training CARROT-Linear...")

        # Train Linear Regression for quality prediction
        self.linear_score = LinearRegression(fit_intercept=self.fit_intercept)
        self.linear_score.fit(embedding_train, quality_train)

        # Train Linear Regression for token count prediction
        self.linear_count = LinearRegression(fit_intercept=self.fit_intercept)
        self.linear_count.fit(embedding_train, token_count_train)

        print("✅ CARROT-Linear training complete")

        if save_dir:
            self.save(save_dir)

    def predict(self, embedding_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict quality scores and token counts.

        Args:
            embedding_test: Test embeddings (n_samples, embedding_dim)

        Returns:
            Tuple of (quality_pred, token_count_pred)
        """
        quality_pred = self.linear_score.predict(embedding_test)
        token_count_pred = self.linear_count.predict(embedding_test)
        return quality_pred, token_count_pred

    def save(self, save_dir: str):
        """Save trained models to directory."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.linear_score, f"{save_dir}/linear_score.joblib")
        joblib.dump(self.linear_count, f"{save_dir}/linear_count.joblib")
        print(f"💾 Saved CARROT-Linear models to {save_dir}")

    def load(self, load_dir: str):
        """Load trained models from directory."""
        self.linear_score = joblib.load(f"{load_dir}/linear_score.joblib")
        self.linear_count = joblib.load(f"{load_dir}/linear_count.joblib")


# =============================================================================
# CARROT Router Interface
# =============================================================================

class CarrotRouter:
    """
    CARROT Router for LLM quality and cost prediction.

    Provides clean interfaces for predicting quality scores and token counts
    for different LLMs given query embeddings.
    """

    def __init__(self,
                 model_dir: str,
                 model_type: str = 'knn',
                 encoder_model: str = None):
        """
        Initialize CARROT Router.

        Args:
            model_dir: Directory containing trained CARROT models and metadata
            model_type: Type of model to use ('knn' or 'linear')
            encoder_model: Sentence transformer model name (if None, loads from metadata)
        """
        self.model_dir = Path(model_dir)
        self.model_type = model_type.lower()

        if self.model_type not in ['knn', 'linear']:
            raise ValueError(f"model_type must be 'knn' or 'linear', got '{model_type}'")

        # Load metadata
        metadata_path = self.model_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.model_names = self.metadata['model_names']
        self.name_to_idx = {name: i for i, name in enumerate(self.model_names)}

        # Load encoder
        encoder_name = encoder_model or self.metadata['encoder_model']
        print(f"Loading encoder: {encoder_name}")
        self.encoder = SentenceTransformer(encoder_name)

        # Load CARROT model
        model_subdir = self.model_dir / f"carrot_{self.model_type}"
        if not model_subdir.exists():
            raise FileNotFoundError(f"Model directory not found at {model_subdir}")

        print(f"Loading CARROT-{self.model_type.upper()} from {model_subdir}")
        if self.model_type == 'knn':
            self.carrot_model = CarrotKNNBaseline(load_dir=str(model_subdir))
        else:
            self.carrot_model = CarrotLinearBaseline(load_dir=str(model_subdir))

        print(f"✅ CARROT Router initialized successfully")
        print(f"   Models: {self.model_names}")
        print(f"   Type: {self.model_type.upper()}")

    def encode(self, query: Union[str, List[str]]) -> np.ndarray:
        """
        Encode query text into embeddings.

        Args:
            query: Single query string or list of queries

        Returns:
            Embeddings array of shape (embedding_dim,) for single query
            or (n_queries, embedding_dim) for multiple queries
        """
        single_query = isinstance(query, str)
        if single_query:
            query = [query]

        # MAX_TOKENS = 512

        # def _truncate(txt):
        #     words = txt.split()
        #     if len(words) > MAX_TOKENS:
        #         return " ".join(words[:MAX_TOKENS])
        #     return txt

        # query = [_truncate(q) for q in query]

        embeddings = self.encoder.encode(query, convert_to_numpy=True)

        # Return 1D array for single query
        if single_query:
            return embeddings[0]
        return embeddings

    def get_quality(self,
                    input_embedding: np.ndarray,
                    llm_name: str) -> Union[float, np.ndarray]:
        """
        Get predicted quality score for a specific LLM.

        Args:
            input_embedding: Query embedding(s), shape (embedding_dim,) or (n_queries, embedding_dim)
            llm_name: Name of the LLM model

        Returns:
            Quality score (0.0-1.0) or array of scores if multiple queries
        """
        if llm_name not in self.name_to_idx:
            raise ValueError(f"Unknown LLM '{llm_name}'. Available: {self.model_names}")

        # Ensure 2D array
        if input_embedding.ndim == 1:
            input_embedding = input_embedding.reshape(1, -1)
            single_query = True
        else:
            single_query = False

        # Get predictions for all models
        quality_pred, _ = self.carrot_model.predict(input_embedding)

        # Extract prediction for requested LLM
        llm_idx = self.name_to_idx[llm_name]
        quality_score = quality_pred[:, llm_idx]

        # Return scalar if single query
        if single_query:
            return float(quality_score[0])
        return quality_score

    def get_cost(self,
                 input_embedding: np.ndarray,
                 llm_name: str) -> Union[float, np.ndarray]:
        """
        Get predicted token count (cost) for a specific LLM.

        Args:
            input_embedding: Query embedding(s), shape (embedding_dim,) or (n_queries, embedding_dim)
            llm_name: Name of the LLM model

        Returns:
            Token count or array of counts if multiple queries
        """
        if llm_name not in self.name_to_idx:
            raise ValueError(f"Unknown LLM '{llm_name}'. Available: {self.model_names}")

        # Ensure 2D array
        if input_embedding.ndim == 1:
            input_embedding = input_embedding.reshape(1, -1)
            single_query = True
        else:
            single_query = False

        # Get predictions for all models
        _, cost_pred = self.carrot_model.predict(input_embedding)

        # Extract prediction for requested LLM
        llm_idx = self.name_to_idx[llm_name]
        token_count = cost_pred[:, llm_idx]

        # Return scalar if single query
        if single_query:
            return float(token_count[0])
        return token_count

    def get_quality_all(self, input_embedding: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """Get predicted quality scores for all LLMs."""
        if input_embedding.ndim == 1:
            input_embedding = input_embedding.reshape(1, -1)
            single_query = True
        else:
            single_query = False

        quality_pred, _ = self.carrot_model.predict(input_embedding)

        results = {}
        for llm_name, idx in self.name_to_idx.items():
            scores = quality_pred[:, idx]
            results[llm_name] = float(scores[0]) if single_query else scores

        return results

    def get_cost_all(self, input_embedding: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """Get predicted token counts (costs) for all LLMs."""
        if input_embedding.ndim == 1:
            input_embedding = input_embedding.reshape(1, -1)
            single_query = True
        else:
            single_query = False

        _, cost_pred = self.carrot_model.predict(input_embedding)

        results = {}
        for llm_name, idx in self.name_to_idx.items():
            counts = cost_pred[:, idx]
            results[llm_name] = float(counts[0]) if single_query else counts

        return results

    def predict_from_text(self,
                         query: Union[str, List[str]],
                         llm_name: str = None) -> Dict[str, Union[float, np.ndarray, Dict]]:
        """Convenience method: encode text and predict quality + cost."""
        embeddings = self.encode(query)

        if llm_name is not None:
            quality = self.get_quality(embeddings, llm_name)
            cost = self.get_cost(embeddings, llm_name)
            return {'quality': quality, 'cost': cost}
        else:
            quality = self.get_quality_all(embeddings)
            cost = self.get_cost_all(embeddings)
            return {'quality': quality, 'cost': cost}

    @property
    def available_models(self) -> List[str]:
        """Get list of available LLM names."""
        return self.model_names.copy()

    def __repr__(self):
        return (f"CarrotRouter(type={self.model_type.upper()}, "
                f"n_models={len(self.model_names)}, "
                f"encoder={self.metadata['encoder_model']})")


def load_carrot_router(model_dir: str, model_type: str = 'knn') -> CarrotRouter:
    """
    Convenience function to load a CARROT router.

    Args:
        model_dir: Directory containing trained CARROT models
        model_type: 'knn' or 'linear'

    Returns:
        Initialized CarrotRouter instance
    """
    return CarrotRouter(model_dir=model_dir, model_type=model_type)


# =============================================================================
# Routing Utilities
# =============================================================================

def route_baseline(Y_hat_score: np.ndarray, Y_hat_count: np.ndarray,
                   Y_score_true: np.ndarray, Y_count_true: np.ndarray,
                   lamb_range: np.ndarray, sizes_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute routing performance across cost-quality tradeoff parameter lambda.

    Args:
        Y_hat_score: Predicted quality scores (n_samples, n_models)
        Y_hat_count: Predicted token counts (n_samples, n_models)
        Y_score_true: True quality scores (n_samples, n_models)
        Y_count_true: True token counts (n_samples, n_models)
        lamb_range: Array of lambda values for cost-quality tradeoff
        sizes_vec: Model size vector (n_models,)

    Returns:
        Tuple of (router_cost, router_perf):
            - router_cost: Average cost at each lambda
            - router_perf: Average quality at each lambda
    """
    n_samples = Y_hat_score.shape[0]
    router_cost = []
    router_perf = []

    for lamb in lamb_range:
        # Compute routing score: (1 - λ) * quality - λ * cost
        routing_scores = (1 - lamb) * Y_hat_score - lamb * (Y_hat_count * sizes_vec[None, :])

        # Select best model for each sample
        selected_models = np.argmax(routing_scores, axis=1)

        # Compute actual cost and performance
        actual_costs = Y_count_true[np.arange(n_samples), selected_models]
        actual_quality = Y_score_true[np.arange(n_samples), selected_models]

        router_cost.append(actual_costs.mean())
        router_perf.append(actual_quality.mean())

    return np.array(router_cost), np.array(router_perf)
