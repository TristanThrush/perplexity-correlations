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
from custom_evals.jeopardy import jeopardy
from custom_evals.jeopardy_acc import jeopardy_accuracy
import math
from tokenization_utils import batch_tokenize_with_percentage_based_indices, compute_average_loss_from_indices, batch_tokenize_with_char_step, batch_tokenize_with_token_str_info, compute_average_loss_from_index_tuples, compute_token_loss_dicts

custom_evals = {"jeopardy": jeopardy, "jeopardy_accuracy": jeopardy_accuracy}

parser = argparse.ArgumentParser()

parser.add_argument("--config")

parser.add_argument("--hf_llm_name", required=False)
parser.add_argument("--hf_llm_family", required=False)
parser.add_argument("--eleuther_eval_names", nargs="*", required=False)
parser.add_argument("--eleuther_eval_metrics", nargs="*", required=False)
parser.add_argument("--eleuther_eval_lower_is_better", nargs="*", required=False)
parser.add_argument("--eleuther_eval_num_fewshot", nargs="*", required=False)
parser.add_argument("--custom_evals", nargs="*", required=False)
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
parser.add_argument("--hf_llm_batch_size", type=int, default=1)
parser.add_argument("--sub_chunk_char_step", type=int, default=10)  # Small numbers here are disk space intensive, so not reccomended for a large number of documents.
parser.add_argument("--mode", default="suffix")  # suffix, token, sub_chunk

args = parser.parse_args()

if args.mode == "suffix":
    percentage_positions_list = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    if args.hf_llm_name is not None and "AI-Sweden-Models" in args.hf_llm_name:  # REEEEEE
        percentage_positions_list = []
else:
    percentage_positions_list = []

if args.chunked_pretraining_data_sample == "None":
    args.chunked_pretraining_data_sample = None

# If args.config is specified, use this script just to kick off a bunch
# of jobs, and then exit from the script
if args.config is not None:
    with open(args.config, "r") as file:
        config = SimpleNamespace(**yaml.safe_load(file))

    mode = getattr(config, "mode", "suffix")
        
    eleuther_eval_names = []
    eleuther_eval_metrics = []
    eleuther_eval_num_fewshot = []
    eleuther_eval_lower_is_better = []
    for eval in config.evals:
        eval = SimpleNamespace(**eval)
        eleuther_eval_names.append(eval.eleuther_name)
        eleuther_eval_metrics.append(eval.metric)
        eleuther_eval_lower_is_better.append(eval.lower_is_better)
        eleuther_eval_num_fewshot.append(eval.num_fewshot)
    eleuther_eval_names = " ".join(eleuther_eval_names)
    eleuther_eval_metrics = " ".join(eleuther_eval_metrics)
    eleuther_eval_num_fewshot = " ".join(
        [str(obj) for obj in eleuther_eval_num_fewshot]
    )
    eleuther_eval_lower_is_better = " ".join(
        [str(obj) for obj in eleuther_eval_lower_is_better]
    )

    custom_evals = " ".join(config.custom_evals)

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
'{eleuther_eval_metrics}' '{eleuther_eval_num_fewshot}' \
'{eleuther_eval_lower_is_better}' '{config.chunked_pretraining_data_sample}' \
'{config.error_output_csv}' '{config.bpb_output_csv_prefix}' '{custom_evals}' '{mode}'"
                subprocess.call(command, shell=True)
    sys.exit()


if None in (
    args.hf_llm_family,
    args.hf_llm_name,
    args.eleuther_eval_names,
    args.eleuther_eval_metrics,
    args.eleuther_eval_num_fewshot,
    args.eleuther_eval_lower_is_better,
    args.error_output_csv,
    args.bpb_output_csv_prefix,
):
    parser.error(
        "Arguments:\n\
--hf_llm_name\n\
--eleuther_eval_names\n\
--eleuther_eval_metrics\n\
--eleuther_eval_num_fewshot\n\
--eleuther_eval_lower_is_better\n\
--error_output_csv\n\
--bpb_output_csv_prefix\n\
are required if --config is not provided."
    )

os.makedirs(args.raw_job_output_path, exist_ok=True)

if args.chunked_pretraining_data_sample is not None:
    ds = load_from_disk(args.chunked_pretraining_data_sample)

try:
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_llm_name,
        revision=args.hf_llm_revision,
        trust_remote_code=True,
        use_fast=True,
    )
    print("Loaded fast tokenizer successfully!")
except ValueError:
    print("Fast tokenizer not available, falling back to slow tokenizer.")
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

    if args.mode == "suffix":
        inputs, suffix_indices, char_indices_list = batch_tokenize_with_percentage_based_indices(tokenizer, texts, percentage_positions_list)
    elif args.mode == "token":
        inputs, token_strings = batch_tokenize_with_token_str_info(tokenizer, texts)
    else:  # "sub_chunk"
        inputs, step_indices, char_indices_list = batch_tokenize_with_char_step(tokenizer, texts, args.sub_chunk_char_step)
        
    inputs.to(args.device)

    # Some models require this.
    inputs["attention_mask"] = inputs["attention_mask"].bool()

    max_len = model.config.max_position_embeddings if (hasattr(model, "config") and hasattr(model.config, "max_position_embeddings")) else None
    if max_len is None: 
        max_len = tokenizer.model_max_length if hasattr(tokenizer, "model_max_length") else None

    # UGH OPT WHYYY REEE
    if 'opt-2.7b' in args.hf_llm_name:
        max_len = 1024+512

    if max_len is None or len(inputs["input_ids"][0]) <= max_len:
        try:
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
            
            if args.mode == "suffix":
                # This averages while ignoring the padding
                losses = loss.sum(dim=1) / inputs.attention_mask[..., 1:].sum(dim=1)
                suffix_losses = compute_average_loss_from_indices(loss, suffix_indices, inputs.attention_mask)
            elif args.mode == "token":
                token_loss_dicts = compute_token_loss_dicts(loss, inputs.attention_mask, token_strings)
            else:  # "sub_chunk"
                step_losses_list = compute_average_loss_from_index_tuples(loss, step_indices, inputs.attention_mask)
 
        except Exception as e:
            print(e)
            if args.mode == "suffix":
                losses = torch.full((args.hf_llm_batch_size,), float('nan'))
                suffix_losses = torch.full((args.hf_llm_batch_size,len(percentage_positions_list)), float('nan'))
            elif args.mode == "token":
                token_loss_dicts = [{} for _ in range(args.hf_llm_batch_size)]
            else:
                step_losses_list = [[float('nan') for _ in range(len(tuples))] for tuples in step_indices]

    else:     
        if args.mode == "suffix":
            losses = torch.full((args.hf_llm_batch_size,), float('nan'))
            suffix_losses = torch.full((args.hf_llm_batch_size,len(percentage_positions_list)), float('nan'))
        elif args.mode == "token":
            token_loss_dicts = [{} for _ in range(args.hf_llm_batch_size)]
        else:
            step_losses_list = [[float('nan') for _ in range(len(tuples))] for tuples in step_indices]

    if args.mode == "suffix":
        output_examples = {
            "id": examples["id"],
            "chunk": examples["chunk"],
            "loss": losses.tolist(),
            "token_count": inputs.attention_mask.sum(dim=1).tolist(),
            "byte_count": [len(text.encode("utf-8")) for text in texts], 
        }

        for index, item in enumerate(percentage_positions_list):
            output_examples[f"loss_{str(item)}_percent_prefix"] = suffix_losses[:,index].tolist()
            output_examples[f"token_count_{str(item)}_percent_prefix"] = inputs.attention_mask[:,suffix_indices[:,index]:].sum(dim=1).tolist() 
            output_examples[f"byte_count_{str(item)}_percent_prefix"] = [len(text[char_indices[index]:].encode("utf-8")) for text, char_indices in zip(texts, char_indices_list)]
    

        if "domain" in examples.keys():
            output_examples["domain"] = examples["domain"]
    elif args.mode == "token":
        output_examples = {
            "id": [],
            "chunk": [],
            "loss": [],
            "token_count": [],
            "byte_count": [],
        }
        if "domain" in examples.keys():
            output_examples["domain"] = []
        for index, token_loss_dict in enumerate(token_loss_dicts):
            output_examples["id"] += [examples["id"][index]]*len(token_loss_dict)
            for token_str, (loss, token_count) in token_loss_dict.items():
                output_examples["chunk"].append(str(examples["chunk"][index]) + "_" + token_str)
                output_examples["loss"].append(loss)
                output_examples["token_count"].append(token_count)
                output_examples["byte_count"].append(len(token_str.encode("utf-8")))
                if "domain" in examples.keys():
                    output_examples["domain"].append(examples["domain"][index])
    else:  # "sub_chunk" 
        output_examples = {
            "id": [],
            "chunk": [],
            "loss": [],
            "token_count": [],
            "byte_count": [],
        }
        if "domain" in examples.keys():
            output_examples["domain"] = []
        for index, char_indices in enumerate(char_indices_list):
            output_examples["id"] += [examples["id"][index]]*len(char_indices)
            for loss_index, start_index in enumerate(char_indices):
                output_examples["chunk"].append(str(examples["chunk"][index]) + "_" + str(start_index) + ":" + str(start_index + args.sub_chunk_char_step))
                output_examples["loss"].append(step_losses_list[index][loss_index])
                output_examples["byte_count"].append(len(texts[index][start_index:start_index + args.sub_chunk_char_step].encode("utf-8")))
                output_examples["token_count"].append(inputs.attention_mask[index][step_indices[index][loss_index][0]:step_indices[index][loss_index][1]].sum(dim=0))
                if "domain" in examples.keys():
                    output_examples["domain"].append(examples["domain"][index])

    return output_examples


if args.chunked_pretraining_data_sample is not None:
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

            print("NaNs in shard: ", sum(1 for item in shard['loss'] if math.isnan(item)))

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


def aggregate_by_domain_or_id(df, agg_groups, percent_prefix_designation=""):
    keep_columns = ["id", "token_count" + percent_prefix_designation, "byte_count" + percent_prefix_designation, "loss" + percent_prefix_designation]
    if "domain" in df.columns:
        keep_columns.append("domain")
    if "chunk" in df.columns:
        keep_columns.append("chunk")
    df = df[keep_columns].copy()
    result = df.dropna(axis=0, how="any")
    result = (
        result.groupby(agg_groups)
        .agg(
            loss=(
                "loss" + percent_prefix_designation,
                lambda x: weighted_mean(result.loc[x.index], "loss" + percent_prefix_designation, "token_count" + percent_prefix_designation),
            ),
            token_count=("token_count" + percent_prefix_designation, "sum"),
            byte_count=("byte_count" + percent_prefix_designation, "sum"),
        )
        .reset_index()
    )
    return result

def get_bpb(df, percent_prefix_designation=""): 
    keep_columns = ["token_count" + percent_prefix_designation, "byte_count" + percent_prefix_designation, "loss" + percent_prefix_designation]
    if "domain" in df.columns:
        keep_columns.append("domain")
    if "chunk" in df.columns:
        keep_columns.append("chunk")
    if "id" in df.columns:
        keep_columns.append("id")
        df["id"] = df["id"].astype(str)
    df = df[keep_columns].copy()

    df[new_column_name] = (
        (df["token_count" + percent_prefix_designation] / df["byte_count" + percent_prefix_designation]) * df["loss" + percent_prefix_designation] / np.log(2)
    )
    df.drop(columns=["token_count" + percent_prefix_designation, "byte_count" + percent_prefix_designation, "loss" + percent_prefix_designation], inplace=True)
    
    return df

bpb_df_dict = {}
if args.chunked_pretraining_data_sample is not None:
    for percent_prefix_designation in [""] + [f"_{str(item)}_percent_prefix" for item in percentage_positions_list]:
        bpb_dfs = [get_bpb(loss_df, percent_prefix_designation)]

        if "domain" in loss_df.columns:
            agg_groups = [["chunk", "id", "domain"], ["id", "domain"], ["domain"]]
        else:
            agg_groups = [["chunk", "id"], ["id"]]

        for agg_group in agg_groups[1:]:
            bpb_dfs.append(get_bpb(aggregate_by_domain_or_id(loss_df, agg_group, percent_prefix_designation)))

        bpb_df_dict[percent_prefix_designation] = bpb_dfs


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
                if "id" in shared_df.columns:
                    shared_df["id"] = shared_df["id"].astype(str)
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


if args.chunked_pretraining_data_sample is not None:
    for percent_prefix_designation, bpb_dfs in bpb_df_dict.items():
        for index in range(len(agg_groups)):
            bpb_df = bpb_dfs[index]
            agg_group = agg_groups[index]
            bpb_output_csv_name = f"{args.bpb_output_csv_prefix}_{agg_group[0]}{percent_prefix_designation}.csv"
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

error_dict = {
    "benchmark": args.custom_evals + [f"{name}_{shots}" for name, shots in zip(args.eleuther_eval_names, args.eleuther_eval_num_fewshot)],
    new_column_name: [custom_evals[custom_eval](model, tokenizer, args.device) for custom_eval in args.custom_evals],
}

for index in range(len(args.eleuther_eval_names)):
    
    if args.eleuther_eval_num_fewshot[index] is not None:
        results = lm_eval.simple_evaluate(
            model=hflm_eleuther,
            tasks=[args.eleuther_eval_names[index]],
            batch_size="auto",
            limit=5000,
            bootstrap_iters=1000,
            log_samples=False,
            num_fewshot=int(args.eleuther_eval_num_fewshot[index]),
        )
    else:
        results = lm_eval.simple_evaluate(
            model=hflm_eleuther,
            tasks=[args.eleuther_eval_names[index]],
            batch_size="auto",
            limit=5000,
            bootstrap_iters=1000,
            log_samples=False,
        )
    print(results)

    name = args.eleuther_eval_names[index]
    metric = args.eleuther_eval_metrics[index]
    lower_is_better = ast.literal_eval(args.eleuther_eval_lower_is_better[index])
    score = results["results"][name][metric]
    #if not lower_is_better:
    #    score = 1 - score
    error_dict[new_column_name].append(score)

error_df = pd.DataFrame.from_dict(error_dict)
error_lock_file_pathname = get_lockfile_pathname(args.error_output_csv)
update_csv_async(
    args.error_output_csv, error_lock_file_pathname, error_df, ["benchmark"]
)
