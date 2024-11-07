import fasttext
from datasets import load_from_disk
import numpy as np
import ast
import argparse
from types import SimpleNamespace
import yaml
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()

with open(args.config, "r") as file:
    config = SimpleNamespace(**yaml.safe_load(file))

os.makedirs(config.output_dir, exist_ok=True)

for target in config.targets:
    fasttext_model_path = target["fasttext_model_path"]
    output_name = target["output_name"]

    model = fasttext.load_model(fasttext_model_path) #'openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin')
    total_labels = len(model.get_labels())

    ds = load_from_disk(config.hf_dataset)[config.split]

    # Run fasttext high-quality (hq) classifier
    def classify_text(example):
        text = example[config.text_column].replace("\n", " ")
        labels, probabilities = model.predict(text, k=total_labels)
        if '__label__hq' in labels:
            return probabilities[labels.index('__label__hq')]
        else:
            return probabilities[labels.index('__label__include')]

    ds = ds.map(lambda example: {"fasttext_hq_prob": classify_text(example), "doc_id": example[config.id_column]}, remove_columns=ds.column_names, num_proc=64)

    # Build a dict of doc id to fasttext hq prob and domain (domain is only included here for debugging)
    doc_id_to_fasttext_hq_prob = {}
    def build_fasttext_dict(example):
        doc_id_to_fasttext_hq_prob[example["doc_id"]] = example["fasttext_hq_prob"]

    ds.map(build_fasttext_dict) 

    # Get page name to index in the ordered doc ids for pretraining
    page_name_to_index = {}
    ordered_page_names = np.load(config.file_prefix + "_id.npy")
    for i, doc_id in enumerate(ordered_page_names):
        page_name_to_index[doc_id] = i

    # Create sampling distribution by iteratively including the highest hq prob pages until we match or exceed the desired number of tokens
    ordered_token_counts = np.load(config.file_prefix + "_len.npy")
    page_names_sorted_by_fasttext_hq_prob = sorted(doc_id_to_fasttext_hq_prob.items(), key=lambda item: item[1], reverse=True)
    print("highest hq prob entries:", page_names_sorted_by_fasttext_hq_prob[:20])
    print("lowest hq prob entries:", page_names_sorted_by_fasttext_hq_prob[-20:])

    current_token_count = 0
    num_included_pages = 0
    sampling_wt = np.zeros(ordered_page_names.shape)
    for doc_id, _ in page_names_sorted_by_fasttext_hq_prob:
        if doc_id in page_name_to_index:
            doc_token_count = ordered_token_counts[page_name_to_index[doc_id]]
            sampling_wt[page_name_to_index[doc_id]] = doc_token_count
            current_token_count += doc_token_count
            num_included_pages += 1
            if current_token_count >= config.desired_token_count:
                print(f"created sampling wt. Num included pages={num_included_pages}, Num tokens={current_token_count}")
                break

    sampling_wt /= sampling_wt.sum()

    np.save(os.path.join(config.output_dir, output_name), sampling_wt)
