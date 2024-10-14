import yaml
from types import SimpleNamespace
import argparse
from transformers import AutoTokenizer
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
from perplexity_correlations.projection import (
    linear
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


def get_X(df):
    agg_columns = [col for col in ["domain", "chunk", "id"] if col in df.columns]
    ordered_columns = df[agg_columns]
    df = df.drop(columns=agg_columns)
    df = df.sort_index(axis=1)
    X_df = df.T
    return X_df, ordered_columns


def get_y(df, target_benchmarks):
    df = df[df["benchmark"].isin(target_benchmarks)]
    df = df.sort_index(axis=1)
    y_df = df.mean(numeric_only=True)
    return y_df


parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = SimpleNamespace(**yaml.safe_load(file))

estimator = estimators[config.estimator]
X_df, labels_df = get_X(pd.read_csv(config.bpb_csv))

ds = load_from_disk(config.chunked_pretraining_data_sample)

tokenizer = AutoTokenizer.from_pretrained(config.hf_tokenizer_name, clean_up_tokenization_spaces=True)
ds = ds.map(lambda example: {"token_count": len(tokenizer(example["text"], truncation=False, padding=False)["input_ids"])}, num_proc=config.num_proc)

aggregation_columns = [
    column for column in ["id", "chunk", "domain"] if column in labels_df.columns
]

token_count_df = ds.to_pandas()
token_count_df = token_count_df.groupby(aggregation_columns, as_index=False).sum(numeric_only=True)

ordered_token_counts = pd.merge(
    labels_df,
    token_count_df,
    on=aggregation_columns,
    how="left",
)["token_count"].to_numpy()

thresholds = (ordered_token_counts/ordered_token_counts.sum())*(1/config.desired_filter_ratio)

ds = ds.train_test_split(test_size=0.05)

for group in config.target_benchmark_groups:
    group = SimpleNamespace(**group)

    y_df = get_y(pd.read_csv(config.error_csv), group.benchmarks)

    X_df = X_df.dropna(axis=0)
    y_df = y_df.dropna(axis=0)

    common_index = y_df.index.intersection(X_df.index)

    # Reindex both dataframes to keep only the common models and align the order
    y_df = y_df.reindex(common_index)
    X_df = X_df.reindex(common_index)

    y = y_df.to_numpy()
    X = X_df.to_numpy()
    print("X dim:", X.shape)

    estimate = estimator(X, y)

    projected_estimate = linear(estimate, thresholds)

    labels = np.array(["__label__exclude"] * len(projected_estimate))
    labels[np.nonzero(projected_estimate)] = "__label__include"

    labels_df["label"] = labels

    def fasttext_label_aggregation(col, col_name):
        if col_name == "text":
            return "".join(col)
        elif pd.api.types.is_string_dtype(col):
            return col.iloc[0]
        elif pd.api.types.is_numeric_dtype(col):
            return col.sum()
        else:
            return None

    train_df = ds["train"].to_pandas()
    train_df = pd.merge(
        train_df,
        labels_df,
        on=aggregation_columns,
        how="inner",
    )
    train_df = train_df.groupby(config.fasttext_label_aggregation, as_index=False).apply(lambda group: group.apply(lambda col: fasttext_label_aggregation(col, col.name)), include_groups=False)
    train_df = train_df[["label", "text", "token_count"]]

    test_df = ds["test"].to_pandas()
    test_df = pd.merge(
        test_df,
        labels_df,
        on=aggregation_columns,
        how="inner",
    )
    test_df = test_df.groupby(config.fasttext_label_aggregation, as_index=False).apply(lambda group: group.apply(lambda col: fasttext_label_aggregation(col, col.name)), include_groups=False)
    test_df = test_df[["label", "text", "token_count"]]

    os.makedirs("fasttext_datasets", exist_ok=True)

    # Save the processed data to a file
    train_df[["label", "text"]].to_csv(
        f"fasttext_datasets/{group.name}.train",
        index=False,
        sep=" ",
        header=False,
    )

    # Train the FastText model
    model = fasttext.train_supervised(
        input=f"fasttext_datasets/{group.name}.train", wordNgrams=2
    )
    
    os.makedirs("fasttext_models", exist_ok=True)
    # Save the model
    model.save_model(f"fasttext_models/{group.name}.bin")


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

    test_results["total_language_dist"] = (test_df.groupby("language")["token_count"].sum() / test_df["token_count"].sum()).to_dict()

    test_df_selected = test_df[test_df["pred"] == "__label__include"]
    test_results["selected_language_dist"] = (test_df_selected.groupby("language")["token_count"].sum() / test_df_selected["token_count"].sum()).to_dict()

    test_df_gold_selected = test_df[test_df["label"] == "__label__include"]
    test_results["gold_language_dist"] = (test_df_gold_selected.groupby("language")["token_count"].sum() / test_df_gold_selected["token_count"].sum()).to_dict()
    
    os.makedirs("fasttext_info", exist_ok=True)
    with open(f"fasttext_info/{group.name}.yml", "w") as file:
        yaml.dump(test_results, file)
