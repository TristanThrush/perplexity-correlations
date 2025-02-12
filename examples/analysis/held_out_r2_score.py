import yaml
from types import SimpleNamespace
import argparse
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from perplexity_correlations.estimation import (
    product,
    sign,
    sign_cdf,
    sign_sign,
    spearmanr,
)
from perplexity_correlations.projection import linear
import numpy as np
import os
from datasets import load_from_disk

estimators = {
    "product": product,
    "sign": sign,
    "sign_cdf": sign_cdf,
    "sign_sign": sign_sign,
    "spearmanr": spearmanr,
}


def get_X(df):
    agg_columns = [col for col in ["domain", "chunk", "id"] if col in df.columns]
    ordered_columns = df[agg_columns]
    df = df.drop(columns=agg_columns)
    df = df.sort_index(axis=1)
    X_df = df.T
    print(X_df)
    return X_df, ordered_columns


def get_y(df, target_benchmarks):
    df = df[df["benchmark"].isin(target_benchmarks)]
    print(df)
    df = df.sort_index(axis=1)
    y_df = df.mean(numeric_only=True)
    return y_df


parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = SimpleNamespace(**yaml.safe_load(file))

estimator = estimators[config.estimator]

test_results = {}

if config.display_top_chunks and not config.chunk_is_token:
    chunk_to_text = {}
    ds = datasets.load_from_disk(config.chunked_pretraining_dataset)
    def build_chunk_to_text(example):
        chunk_to_text[(example["id"], example["chunk"])] = example["text"]
    ds = ds.map(build_chunk_to_text)

for group in config.target_benchmark_groups:
    group = SimpleNamespace(**group)
    X_df, labels_df = get_X(pd.read_csv(group.bpb_csv))
    
    aggregation_columns = [
        column for column in ["id", "chunk", "domain"] if column in labels_df.columns
    ]

    y_df = get_y(pd.read_csv(config.error_csv), group.benchmarks)

    #X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #y_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #X_df = X_df.dropna(axis=0)
    #y_df = y_df.dropna(axis=0)
    #X_df = X_df.fillna(0)
    print("X dim:", X_df)

    common_index = y_df.index.intersection(X_df.index)

    # Reindex both dataframes to keep only the common models and align the order
    y_df = y_df.reindex(common_index)
    X_df = X_df.reindex(common_index)

    y = y_df.to_numpy()
    X = X_df.to_numpy()
    print("X dim:", X.shape)

    estimate = estimator(X, y)
    estimate = np.nan_to_num(estimate, nan=0, posinf=0, neginf=0)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Just computing this as something interesting to know
    ten_largest_indices = np.argpartition(estimate, -10)[-10:]
    top_ten_from_estimate = (
        labels_df[aggregation_columns].loc[ten_largest_indices].to_dict()
    )

    # TODO include projected estimate
    #projected_estimate = linear(estimate, thresholds)

    # Just computing this as something interesting to know
    '''
    projected_ten_largest_indices = np.argpartition(projected_estimate, -10)[-10:]
    top_ten_from_projected_estimate = (
        labels_df[aggregation_columns].loc[projected_ten_largest_indices].to_dict()
    )
    '''

    print("estimate:", estimate)

    test_results[f"{group.name}_top_ten_from_estimate"] = top_ten_from_estimate
    #test_results[f"{group.name}_top_ten_from_projected_estimate"] = top_ten_from_projected_estimate

    if config.display_top_chunks:
        #if config.chunk_is_token:
        print(top_ten_from_estimate)
        crash
    
    y_ranks = np.argsort(np.argsort(y, axis=0), axis=0) + 1
    y_pred_ranks = np.argsort(np.argsort(X @ estimate, axis=0), axis=0) + 1
    test_results[f"{group.name}_r2"] = r2_score(y_ranks, y_pred_ranks)
    
    # Create 5-fold cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Loop through each split
    cross_validation_r2 = 0
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        estimate = estimator(X_train, y_train)
        y_ranks = np.argsort(np.argsort(y_test, axis=0), axis=0) + 1
        y_pred_ranks = np.argsort(np.argsort(X_test @ estimate, axis=0), axis=0) + 1
        cross_validation_r2 += r2_score(y_ranks, y_pred_ranks)

    cross_validation_r2 /= n_splits
    test_results[f"{group.name}_cross_validation_r2"] = cross_validation_r2

os.makedirs("held_out_r2_results", exist_ok=True)

with open(f"held_out_r2_results/{config.name}.yml", "w") as file:
    yaml.dump(test_results, file)
