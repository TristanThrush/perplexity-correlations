from datasets import load_dataset, Features, Value
from transformers import AutoTokenizer
from ast import literal_eval
from tqdm import tqdm
import yaml
import argparse
from types import SimpleNamespace
import numpy as np
from collections import defaultdict, Counter
import random

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = SimpleNamespace(**yaml.safe_load(file))

ds = load_dataset(
    config.hf_name,
    name=config.subset,
    split=config.split,
)
if config.subsample_ratio < 1:
    sample_size = int(config.subsample_ratio * len(ds))
    ds = ds.select(range(sample_size))

print("full dataset length:", len(ds))

if config.look_in_metadata_for_id:
    ds = ds.map(
        lambda x: {
            config.id_column: literal_eval(x[config.metadata_column])[config.id_column]
        },
        num_proc=config.num_proc,
    )

if config.domain_column is not None:
    if config.look_in_metadata_for_domain:
        ds = ds.map(
            lambda x: {
                config.domain_column: literal_eval(x[config.metadata_column])[
                    config.domain_column
                ]
            },
            num_proc=config.num_proc,
        )
    ds = ds.rename_column(config.domain_column, "domain")


if config.enforce_pages_per_domain:
    domain_counts = Counter(ds["domain"])

    # Filter out domains without enough pages.
    def filter_rows(row):
        if row["domain"] is None:
            return False
        return domain_counts[row["domain"]] >= config.pages_per_domain

    ds = ds.filter(filter_rows, num_proc=config.num_proc)
    print(
        f"number of examples with at least {config.pages_per_domain} domains: ", len(ds)
    )

    # Now, randomly select domains to remove s.t. there are exactly
    # config.pages_per_domain pages from each domain.
    domain_indices = defaultdict(list)
    for i, row in tqdm(enumerate(ds)):
        domain_indices[row["domain"]].append(i)

    balanced_indices = []
    for indices in domain_indices.values():
        balanced_indices.extend(
            np.random.choice(indices, size=config.pages_per_domain, replace=False)
        )

    ds = ds.select(balanced_indices)

    unique_domains_count = len(set(ds["domain"]))
    print("unique domains:", unique_domains_count)


reference_tokenizer = AutoTokenizer.from_pretrained(config.reference_tokenizer_hf_name)


def get_text_chunks_with_reference_tokenizer(examples):
    # We assume the batch size is 1
    text = examples[config.text_column][0]

    reference_tokens = reference_tokenizer(text, return_offsets_mapping=True)

    seq_len = len(reference_tokens.input_ids)

    previous_text_split_index = 0

    text_chunks = []

    for begin_loc in range(0, seq_len, config.reference_tokenizer_chunk_size):
        end_loc = min(begin_loc + config.reference_tokenizer_chunk_size, seq_len - 1)

        text_split_index = reference_tokens.offset_mapping[end_loc][1]
        text_chunk = text[previous_text_split_index:text_split_index]
        text_chunks.append(text_chunk)
        previous_text_split_index = text_split_index

        if end_loc == seq_len - 1:
            break

    assert "".join(text_chunks) == text

    chunked_examples = {
        "text": text_chunks,
        "id": [examples[config.id_column][0]] * len(text_chunks),
        "chunk": list(range(len(text_chunks))),
    }

    if config.domain_column is not None:
        domain = examples["domain"][0]
        chunked_examples["domain"] = [domain] * len(text_chunks)

    return chunked_examples


feature_dict =  {
        "text": Value("string"),
        "id": Value("string"),
        "chunk": Value("int32"),
        "domain": Value("string"),
    }

if config.domain_column is not None:
    feature_dict["domain"] = Value("string")

features = Features(feature_dict)
ds = ds.map(
    get_text_chunks_with_reference_tokenizer,
    features=features,
    batched=True,
    batch_size=1,
    remove_columns=ds.column_names,
    num_proc=config.num_proc,
)

ds.save_to_disk(config.output_name)
