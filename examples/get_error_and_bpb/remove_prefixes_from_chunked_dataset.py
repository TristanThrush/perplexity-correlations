from datasets import load_from_disk
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset")

args = parser.parse_args()

percentage_positions_list = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]

ds = load_from_disk(args.dataset)

for percent in percentage_positions_list:
    def remove_prefix(text):
        return text[max(0, min(len(text) - 1, int(percent * len(text)))):]
    ds_suffix = ds.map(lambda example: {"text": remove_prefix(example["text"])})
    ds_suffix.save_to_disk(args.dataset + f"_{str(percent)}_prefix_removed")
