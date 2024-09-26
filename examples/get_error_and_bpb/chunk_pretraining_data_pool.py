from datasets import load_dataset, Features, Value
from transformers import AutoTokenizer
from ast import literal_eval
from tqdm import tqdm
import yaml
import argparse
from types import SimpleNamespace
import numpy as np
from collections import defaultdict, Counter


np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = SimpleNamespace(**yaml.safe_load(file))

print(config)
ds = load_dataset(
    config.pretraining_data_pool_hf_name,
    name=config.pretraining_data_pool_subset,
    split=config.pretraining_data_pool_split,
)
print("full dataset length:", len(ds))

if config.look_in_metadata_for_id:
    ds = ds.map(
        lambda x: {
            config.id_column: literal_eval(x[config.metadata_column])[config.id_column]
        }
    )

if config.domain_column is not None:
    if config.look_in_metadata_for_domain:
        ds = ds.map(
            lambda x: {
                config.domain_column: literal_eval(x[config.metadata_column])[
                    config.domain_column
                ]
            }
        )


if config.enforce_pages_per_domain:
    ds = ds.rename_column(config.domain_column, "domain")
    domain_counts = Counter(ds["domain"])

    # Filter out domains without enough pages.
    def filter_rows(row):
        if row["domain"] is None:
            return False
        return domain_counts[row["domain"]] >= config.pages_per_domain

    ds = ds.filter(filter_rows)
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

    unique_domains_count = len(set(ds[config.pages_per_domain]))
    print("unique domains:", unique_domains_count)


reference_tokenizer = AutoTokenizer.from_pretrained(config.reference_tokenizer)


def get_text_chunks_with_reference_tokenizer(examples):
    # We assume the batch size is 1
    text = examples[config.text_column][0]

    reference_tokens = reference_tokenizer(text, return_offsets_mapping=True)

    seq_len = len(reference_tokens.input_ids)

    previous_text_split_index = 0

    text_chunks = []
    text_chunk_reference_token_counts = []

    for begin_loc in range(0, seq_len, config.reference_context_size):
        end_loc = min(begin_loc + config.reference_context_size, seq_len - 1)

        text_split_index = reference_tokens.offset_mapping[end_loc][1]
        text_chunk = text[previous_text_split_index:text_split_index]
        text_chunk_reference_token_counts.append(
            len(reference_tokenizer(text_chunk).input_ids)
        )
        text_chunks.append(text_chunk)
        previous_text_split_index = text_split_index

        if end_loc == seq_len - 1:
            break

    assert "".join(text_chunks) == text

    return {
        config.text_column: text_chunks,
        config.id_column: [
            f"{obj[0]}_{obj[1]}"
            for obj in zip(
                examples[config.id_column] * len(text_chunks), range(len(text_chunks))
            )
        ],
        "reference_token_count": text_chunk_reference_token_counts,
    }


features = Features(
    {
        config.text_column: Value("string"),
        config.id_column: Value("string"),
        "reference_token_count": Value("int32"),
    }
)
ds = ds.map(
    get_text_chunks_with_reference_tokenizer,
    features=features,
    batched=True,
    batch_size=1,
    remove_columns=ds.column_names,
)

ds = ds.rename_column(config.text_column, "text")
ds = ds.rename_column(config.id_column, "id")

columns_to_drop = [
    column for column in ds.column_names if column not in ["text", "id", "domain"]
]
ds = ds.remove_columns(columns_to_drop)

ds.save_to_disk(config.output_name)
