from datasets import load_dataset, load_from_disk
from mmap_utils import tokenize_and_mmap, get_dataset
from types import SimpleNamespace
from transformers import AutoTokenizer
import yaml
import numpy as np
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = SimpleNamespace(**yaml.safe_load(file))

def wrap_dataset_iterator(ds, fields):
    for data in ds:
        yield [data[i] for i in fields]

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(config.hf_tokenizer)
    if config.from_disk:
        ds = load_from_disk(config.ds_path)[config.split]
        ds_subset = ds
    else:
        ds = load_dataset(config.ds_path, config.split)
    os.makedirs(config.output_dir, exist_ok=True)
    file_prefix = os.path.join(config.output_dir, config.output_file_prefix)
    max_tokens = 100*(10**9) # 100 bil tokens

    if config.domain is not None:
        doc_id_to_domain = {}
        def build_doc_id_to_domain(example):
            doc_id_to_domain[example[config.id_column]] = example[config.domain]
        ds_subset.map(build_doc_id_to_domain)
        with open(file_prefix + "_doc_id_to_domain.pkl", 'wb') as f:
            pickle.dump(doc_id_to_domain, f)

    def tokenize_function(example):
        return tokenizer(example[config.text_column], padding="do_not_pad", truncation=False)

    # Apply tokenization and remove original columns
    ds_subset = ds_subset.map(tokenize_function, remove_columns=[name for name in ds_subset.column_names if name != config.id_column], num_proc=config.num_proc)

    tokenize_and_mmap(wrap_dataset_iterator(ds_subset, [config.id_column, 'input_ids']), tokenizer, max_tokens, config.context_length, file_prefix)
    len_vecs = np.load(file_prefix + "_len.npy")
    prob_vec = len_vecs / np.sum(len_vecs)
    dataset = get_dataset(prob_vector=prob_vec, ctx_len=ctx_len, memmaped_file=file_prefix + ".mmap", start_map=np.load(file_prefix + "_start.npy"), len_map=np.load(file_prefix + "_len.npy"), max_tokens=max_tokens)

    for i, data in enumerate(dataset):
        print(tokenizer.decode(data["input_ids"]))
        print('--------')
        if i > 10:
            break
