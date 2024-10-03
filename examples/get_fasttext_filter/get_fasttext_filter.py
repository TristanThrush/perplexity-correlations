import yaml
from types import SimpleNamespace
import argparse
from datasets import load_from_disk
import pandas as pd
import fasttext
from perplexity_correlations.estimation import (
    product,
    sign,
    sign_cdf,
    sign_sign,
    spearmanr,
)
import numpy as np
import os


estimators = {
    "product": product,
    "sign": sign,
    "sign_cdf": sign_cdf,
    "sign_sign": sign_sign,
    "spearmanr": spearmanr,
}


def get_X_no_aggregation(df):
    if "domain" in df.columns:
        df = df.drop(columns=["domain"])
    ordered_ids_and_chunks = df[["id", "chunk"]]
    df = df.drop(columns=["id", "chunk"])
    df = df.sort_index(axis=1)
    X = df.to_numpy().T
    return X, ordered_ids_and_chunks


def get_X_id_aggregation(df):
    df = df.groupby("id", as_index=False)
    df = df.mean(numeric_only=True)
    ordered_ids = df[["id"]]
    df = df.drop(columns=["id", "chunk"])
    df = df.sort_index(axis=1)
    X = df.to_numpy().T
    return X, ordered_ids


def get_X_domain_aggregation(df):
    df = df.groupby("domain", as_index=False)
    df = df.mean(numeric_only=True)
    ordered_domains = df[["domain"]]
    df = df.drop(columns=["domain", "chunk"])
    df = df.sort_index(axis=1)
    X = df.to_numpy().T
    return X, ordered_domains


get_X_functions = {
    None: get_X_no_aggregation,
    "id": get_X_id_aggregation,
    "domain": get_X_domain_aggregation,
}


def get_y(df, target_benchmarks):
    df = df[df["benchmark"].isin(["arc_easy", "piqa"])]
    df = df.sort_index(axis=1)
    df = df.mean(numeric_only=True)
    y = df.to_numpy()
    return y


parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = SimpleNamespace(**yaml.safe_load(file))

estimator = estimators[config.estimator]
get_X_function = get_X_functions[config.aggregation]
X, labels_df = get_X_function(pd.read_csv(config.bpb_csv))
y = get_y(pd.read_csv(config.error_csv), config.target_benchmarks)

estimate = estimator(X, y)

# Assume the sample used to generate the BPB data comes from the same
# dist as the data we want to pretrain on. Now, assume that we want to
# pretrain on the best half of the chunks/pages/domains. Because the linear
# projection is not sensitive to the particular values of the estimate
# (only their ranks), we can just take the half of the text with the top
# values in estimate as our pretraining data. We can also train a fastText
# model to distinguish this good pretraining data from other data, which is
# what we do here. You can then use this fastText model as a pretraining
# data filter.

labels = np.array(["__label__exclude"] * len(estimate))
labels[np.argsort(estimate)[int(len(estimate) / 2) :]] = "__label__include"

labels_df["label"] = labels

# Load the training dataset
ds = load_from_disk(config.chunked_pretraining_data_sample)

ds = ds.train_test_split(test_size=0.01)

aggregation_columns = [column for column in ["id", "chunk", "domain"] if column in labels_df.columns]

train_df = ds["train"].to_pandas()
train_df = pd.merge(
    train_df,
    labels_df,
    on=aggregation_columns,
    how="inner",
)
train_df = train_df[["label", "text"]]

test_df = ds["test"].to_pandas()
test_df = pd.merge(
    test_df,
    labels_df,
    on=aggregation_columns,
    how="inner",
)
test_df = test_df[["label", "text"]]

os.makedirs("fasttext_datasets", exist_ok=True)

# Save the processed data to a file
train_df.to_csv(
    f"fasttext_datasets/{config.fasttext_model_output_name}.train", index=False, sep=" ", header=False
)
test_df.to_csv(
    f"fasttext_datasets/{config.fasttext_model_output_name}.valid", index=False, sep=" ", header=False
)

# Train the FastText model
model = fasttext.train_supervised(
    input=f"fasttext_datasets/{config.fasttext_model_output_name}.train", wordNgrams=2
)

# Evaluate the model
result = model.test(f"fasttext_datasets/{config.fasttext_model_output_name}.valid")
print(f"Number of samples: {result[0]}")
print(f"Precision@1: {result[1]}")
print(f"Recall@1: {result[2]}")

# Save the model
model.save_model(f"{config.fasttext_model_output_name}.bin")
