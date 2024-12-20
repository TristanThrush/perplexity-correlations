from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_from_disk, concatenate_datasets
import argparse
import torch
from types import SimpleNamespace
import os
import sys
import json
import lm_eval
import yaml
from lm_eval.models.huggingface import HFLM
import subprocess
import numpy as np
from filelock import FileLock
import ast
import time
import pandas as pd
import warnings

parser = argparse.ArgumentParser()

parser.add_argument("--config")

parser.add_argument("--hf_llm_name", required=False)
parser.add_argument("--hf_llm_family", required=False)
parser.add_argument("--eleuther_eval_names", nargs="*", required=False)
parser.add_argument("--eleuther_eval_metrics", nargs="*", required=False)
parser.add_argument("--eleuther_eval_lower_is_better", nargs="*", required=False)
parser.add_argument("--chunked_pretraining_data_sample", required=False)
parser.add_argument("--raw_job_output_path", required=False)
parser.add_argument("--error_output_csv", required=False)
parser.add_argument("--bpb_output_csv_prefix", required=False)

parser.add_argument("--hf_llm_revision", default="main")
parser.add_argument("--num_loss_shards", type=int, default=50)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--save_model_info", action="store_true")
parser.add_argument("--device", default="cuda")
parser.add_argument("--half_precision", action="store_true")
parser.add_argument("--hf_llm_batch_size", type=int, default=2)

args = parser.parse_args()

# If args.config is specified, use this script just to kick off a bunch
# of jobs, and then exit from the script
if args.config is not None:
    with open(args.config, "r") as file:
        config = SimpleNamespace(**yaml.safe_load(file))

    eleuther_eval_names = []
    eleuther_eval_metrics = []
    eleuther_eval_lower_is_better = []
    for eval in config.evals:
        eval = SimpleNamespace(**eval)
        eleuther_eval_names.append(eval.eleuther_name)
        eleuther_eval_metrics.append(eval.metric)
        eleuther_eval_lower_is_better.append(eval.lower_is_better)
    eleuther_eval_names = " ".join(eleuther_eval_names)
    eleuther_eval_metrics = " ".join(eleuther_eval_metrics)
    eleuther_eval_lower_is_better = " ".join(
        [str(obj) for obj in eleuther_eval_lower_is_better]
    )

    for family in config.llms:
        family = SimpleNamespace(**family)
        for llm in family.hf_names:
            revisions = ["main"]
            if isinstance(llm, dict):
                revisions = llm["revisions"]
                llm = llm["name"]

            for revision in revisions:
                revision_suffix = ""
                if revision != "main":
                    revision_suffix = "_" + revision
                output_path = os.path.join(
                    config.raw_job_output_dir, llm.replace("/", "-") + revision_suffix
                )
                os.makedirs(output_path, exist_ok=True)
                command = f"bash error_and_bpb_scheduler.sh \
'{output_path}' '{family.family}' '{llm}' '{revision}' '{eleuther_eval_names}' \
'{eleuther_eval_metrics}' '{eleuther_eval_lower_is_better}' \
'{config.chunked_pretraining_data_sample}' '{config.error_output_csv}' \
'{config.bpb_output_csv_prefix}'"
                subprocess.call(command, shell=True)
    sys.exit()


if None in (
    args.hf_llm_family,
    args.hf_llm_name,
    args.eleuther_eval_names,
    args.eleuther_eval_metrics,
    args.eleuther_eval_lower_is_better,
    args.chunked_pretraining_data_sample,
    args.error_output_csv,
    args.bpb_output_csv_prefix,
):
    parser.error(
        "Arguments:\n\
--hf_llm_name\n\
--eleuther_eval_names\n\
--eleuther_eval_metrics\n\
--eleuther_eval_lower_is_better\n\
--chunked_pretraining_data_sample\n\
--error_output_csv\n\
--bpb_output_csv_prefix\n\
are required if --config is not provided."
    )

os.makedirs(args.raw_job_output_path, exist_ok=True)

ds = load_from_disk(args.chunked_pretraining_data_sample)

tokenizer = AutoTokenizer.from_pretrained(
    args.hf_llm_name,
    revision=args.hf_llm_revision,
    trust_remote_code=True,
)

if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
    if not hasattr(tokenizer, "eos_token") or tokenizer.eos_token is None:
        tokenizer.pad_token = "<|endoftext|>"
    else:
        tokenizer.pad_token = tokenizer.eos_token

if args.half_precision:
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_llm_name,
        revision=args.hf_llm_revision,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(args.device)
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_llm_name,
        revision=args.hf_llm_revision,
        trust_remote_code=True,
    ).to(args.device)

model.eval()

if args.save_model_info:
    config_dict = AutoConfig.from_pretrained(
        args.hf_llm_name, revision=args.hf_llm_revision, trust_remote_code=True
    ).to_dict()

    info = {}
    info["torch_dtype"] = config_dict.get("torch_dtype", None)
    info["vocab_size"] = config_dict.get("vocab_size", None)
    info["context_size"] = config_dict.get("max_position_embeddings", None)
    info["parameter_count"] = sum(p.numel() for p in model.parameters())

    open(f"{args.raw_job_output_path}/llm_info.json", "w+").write(json.dumps(info))


def get_loss_hf(examples):
    texts = examples["text"]

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to(
        args.device
    )

    # Some models require this.
    inputs["attention_mask"] = inputs["attention_mask"].bool()

    outputs = model(**inputs)

    logits = outputs.logits

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    shift_logits = logits[..., :-1, :].contiguous()

    # Need to set pad indices to -100 for cross entropy loss to ignore the padding.
    pad_indices = torch.where(inputs.attention_mask == 0)
    inputs.input_ids[pad_indices] = -100

    shift_labels = inputs.input_ids[..., 1:].contiguous()

    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    loss = loss.view(shift_labels.size())

    # This averages while ignoring the padding
    losses = loss.sum(dim=1) / inputs.attention_mask[..., 1:].sum(dim=1)

    output_examples = {
        "id": examples["id"],
        "chunk": examples["chunk"],
        "loss": losses.tolist(),
        "token_count": inputs.attention_mask.sum(dim=1).tolist(),
        "byte_count": [len(text.encode("utf-8")) for text in texts],
    }

    if "domain" in examples.keys():
        output_examples["domain"] = examples["domain"]

    return output_examples


# Create a list to hold the shards. This enables us to resume getting the loss
# from the shard where we left off if there is some issue that causes the job to
# exit early.
shards = []

# Shard the dataset and add each shard to the list
for i in range(args.num_loss_shards):
    if args.resume and os.path.exists(f"{args.raw_job_output_path}/loss_shards/{i}"):
        shard = load_from_disk(f"{args.raw_job_output_path}/loss_shards/{i}")
        shards.append(shard)
    else:
        shard = ds.shard(num_shards=args.num_loss_shards, index=i)

        # For efficiency - we want to avoid as much padding as possible
        shard = shard.sort(["reference_token_count"], reverse=[True])

        shard = shard.map(
            lambda example: get_loss_hf(example),
            remove_columns=ds.column_names,
            batched=True,
            batch_size=args.hf_llm_batch_size,
        )

        shard.save_to_disk(f"{args.raw_job_output_path}/loss_shards/{i}")

        shards.append(shard)

loss_df = concatenate_datasets(shards).to_pandas()

# Convert to BPB at the end, so raw losses, token counts, and byte counts are still
# stored in the loss shard datasets in case they would be useful in the future.
# Name the bpb column with the name and family of the LLM, so we can merge it into the
# shared matrix.
revision_suffix = ""
if args.hf_llm_revision != "main":
    revision_suffix = "_" + args.hf_llm_revision
new_column_name = str((args.hf_llm_family, args.hf_llm_name + revision_suffix))


def weighted_mean(df, value_col, weight_col):
    return (df[value_col] * df[weight_col]).sum() / df[weight_col].sum()


def aggregate_by_domain_or_id(df, agg_groups):
    result = df.dropna(axis=0, how="any")
    result = (
        result.groupby(agg_groups)
        .agg(
            loss=(
                "loss",
                lambda x: weighted_mean(result.loc[x.index], "loss", "token_count"),
            ),
            token_count=("token_count", "sum"),
            byte_count=("byte_count", "sum"),
        )
        .reset_index()
    )
    return result


def get_bpb(df):
    df = df.copy()
    df[new_column_name] = (
        (df["token_count"] / df["byte_count"]) * df["loss"] / np.log(2)
    )
    df.drop(columns=["token_count", "byte_count", "loss"], inplace=True)
    return df


bpb_dfs = [get_bpb(loss_df)]

if "domain" in loss_df.columns:
    agg_groups = [["chunk", "id", "domain"], ["id", "domain"], ["domain"]]
else:
    agg_groups = [["chunk", "id"], ["id"]]

for agg_group in agg_groups[1:]:
    bpb_dfs.append(get_bpb(aggregate_by_domain_or_id(loss_df, agg_group)))


# Function to safely read, modify, and write to shared CSV file.
def update_csv_async(
    csv_file_path, lock_file_path, df_to_add, merge_on, lock_timeout=300
):
    # Create a lock for the CSV file
    lock = FileLock(lock_file_path, timeout=lock_timeout)

    try:
        # Acquire the lock
        with lock:
            print(f"Lock acquired by {time.ctime()}")

            # Read the existing CSV file into a DataFrame
            already_added = False
            try:
                shared_df = pd.read_csv(csv_file_path)
                if new_column_name in shared_df.columns:
                    shared_df = shared_df.drop(columns=[new_column_name])
                    warnings.warn(
                        f"{new_column_name} was already in {csv_file_path}. \
Removed original values."
                    )

            except FileNotFoundError:
                # If the CSV doesn't exist yet, just save our matrix
                df_to_add.to_csv(csv_file_path, index=False)
                already_added = True

            if not already_added:
                # Add new data
                shared_df = pd.merge(shared_df, df_to_add, on=merge_on, how="inner")

                # Save the updated DataFrame back to the CSV file
                shared_df.to_csv(csv_file_path, index=False)

            print(f"CSV updated and lock released by {time.ctime()}")
    except TimeoutError:
        print(
            f"Failed to acquire the lock within {lock_timeout} seconds. \
Job is retrying."
        )
        update_csv_async(csv_file_path, lock_file_path, df_to_add, merge_on)


# Now, add this model's BPB to the big shared BPB matrix that all of the jobs are
# creating.
def get_lockfile_pathname(pathname):
    directory, filename = os.path.split(pathname)
    invisible_filename = f".{filename}.lock"
    lockfile_pathname = os.path.join(directory, invisible_filename)
    return lockfile_pathname


for index in range(len(agg_groups)):
    bpb_df = bpb_dfs[index]
    agg_group = agg_groups[index]
    bpb_output_csv_name = f"{args.bpb_output_csv_prefix}_{agg_group[0]}.csv"
    bpb_lock_file_pathname = get_lockfile_pathname(bpb_output_csv_name)
    update_csv_async(
        bpb_output_csv_name,
        bpb_lock_file_pathname,
        bpb_df,
        agg_group,
    )

# Check to see that there are actually evals specified before continuing.
if len(args.eleuther_eval_names) == 0:
    sys.exit()


# Now we evaluate the model on the desired tasks and add the results to the big
# shared eval matrix.
class HFLM_Local(HFLM):
    def get_model_info(self):
        return {}


hflm_eleuther = HFLM_Local(pretrained=model, tokenizer=tokenizer)

results = lm_eval.simple_evaluate(
    model=hflm_eleuther,
    tasks=args.eleuther_eval_names,
    batch_size="auto",
    limit=5000,
    bootstrap_iters=1000,
    log_samples=False,
)

# Make the name of the error column the llm model family and name, so we can
# merge with the big shared error matrix.
error_dict = {
    "benchmark": args.eleuther_eval_names,
    new_column_name: [],
}
for index in range(len(args.eleuther_eval_names)):
    name = args.eleuther_eval_names[index]
    metric = args.eleuther_eval_metrics[index]
    lower_is_better = ast.literal_eval(args.eleuther_eval_lower_is_better[index])
    score = results["results"][name][metric]
    if not lower_is_better:
        score = 1 - score
    error_dict[new_column_name].append(score)

error_df = pd.DataFrame.from_dict(error_dict)
error_lock_file_pathname = get_lockfile_pathname(args.error_output_csv)
update_csv_async(
    args.error_output_csv, error_lock_file_pathname, error_df, ["benchmark"]
)
