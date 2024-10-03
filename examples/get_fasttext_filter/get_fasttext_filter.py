import yaml
from types import SimpleNamespace
import argparse
from datasets import load_from_disk
from huggingface_hub import hf_hub_download
import pandas as pd
import fasttext
from sklearn.metrics import f1_score, precision_score, recall_score
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

ds = ds.train_test_split(test_size=0.05)

aggregation_columns = [
    column for column in ["id", "chunk", "domain"] if column in labels_df.columns
]

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
    f"fasttext_datasets/{config.fasttext_model_output_name}.train",
    index=False,
    sep=" ",
    header=False,
)
test_df.to_csv(
    f"fasttext_datasets/{config.fasttext_model_output_name}.valid",
    index=False,
    sep=" ",
    header=False,
)

# Train the FastText model
model = fasttext.train_supervised(
    input=f"fasttext_datasets/{config.fasttext_model_output_name}.train", wordNgrams=2
)

# Save the model
model.save_model(f"{config.fasttext_model_output_name}.bin")


# Evaluate the model.
test_results = {}


# First, get f1, precision, and recall for classifying the text correctly.
def predict_label(text):
    labels, probabilities = model.predict(text.replace("\n", " "), k=1)
    return labels[0]


test_df["pred"] = test_df["text"].apply(predict_label)
test_results["num_samples"] = len(test_df)
test_results["f1"] = float(f1_score(test_df["label"], test_df["pred"], average="binary", pos_label="__label__include"))
test_results["precision@1"] = float(precision_score(
    test_df["label"], test_df["pred"], average="binary", pos_label="__label__include"
))
test_results["recall@1"] = float(recall_score(
    test_df["label"], test_df["pred"], average="binary", pos_label="__label__include"
))

# Now, a fine-grained eval: see what language(s) our model wants to select.
language_id_model_path = hf_hub_download(
    repo_id="facebook/fasttext-language-identification", filename="model.bin"
)
language_id_model = fasttext.load_model(language_id_model_path)


def predict_language(text):
    labels, probabilities = language_id_model.predict(text.replace("\n", " "), k=1)
    return labels[0]


test_df["language"] = test_df["text"].apply(predict_language)

test_results["total_language_dist"] = (test_df["language"].value_counts() / len(test_df)).to_dict()

test_df_selected = test_df[test_df["pred"] == "__label__include"]
test_results["selected_language_dist"] = (test_df_selected[
    "language"
].value_counts() / len(test_df_selected)).to_dict()

test_df_gold_selected = test_df[test_df["label"] == "__label__include"]
test_results["gold_language_dist"] = (test_df_gold_selected[
    "language"
].value_counts() / len(test_df_gold_selected)).to_dict()

print(test_results)

with open(f"{config.fasttext_model_output_name}_test_results.yml", "w") as file:
    yaml.dump(test_results, file)
