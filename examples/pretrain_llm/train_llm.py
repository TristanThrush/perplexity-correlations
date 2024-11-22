import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
import lm_eval
from lm_eval.models.huggingface import HFLM
import os
import json
import torch
import yaml
import subprocess
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoConfig,
    set_seed,
    DefaultDataCollator,
)

from transformers import Trainer, TrainingArguments, AutoTokenizer
from mmap_utils import get_dataset_async
from peft import LoraConfig, TaskType, get_peft_model
import numpy as np
from types import SimpleNamespace
import wandb


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM']="false"
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_distributed =  world_size > 1
is_main_process = not is_distributed or int(os.environ.get("RANK", 0)) == 0
print("Distributed", is_distributed)

@dataclass
class ScriptArguments:
    """
    Arguments which aren't included in the TrainingArguments
    """
    token_count: Optional[int] = field(
        default=None,
        metadata={"help": "number of pretraining tokens."},
    )
    config: Optional[str] = field(
        default=None,
        metadata={"help": "A config with lots of jobs to kick off"},
    )
    seed: Optional[int] = field(
        default=0,
        metadata={"help": "Training seed."},
    )
    sample_wt_path: str = field(
        default=None,
        metadata={
            "help": "The file path to a numpy sampling weight pickle. Entries should correspond to probability of sampling a document, matching docid"
        },
    )
    mmap_prefix: str = field(
        default=None,
        metadata={
            "help": "The file prefix for the collection of numpy pickle files that encode the red pajama v2 (or other) pretraining dataset. See data_utils.py"
        },
    )
    output_dir_prefix: str = field(
        default="raw_train_llm_outputs",
        metadata={
            "help": "A prefix for the path where models and logs get saved. Combined with a extension-free version of sample_wt_prefix to define the full output path"
        },
    )
    model_id: Optional[str] = field(
        default="EleutherAI/pythia-160m",
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    per_device_train_batch_size: Optional[int] = field(
        default=128,
        metadata={"help": "The Batch Size per GPU used during training"},
    )
    learning_rate: Optional[float] = field(
        default=5e-3, metadata={"help": "Learning Rate for the training"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of gradient accumulation steps to take; artificially increases the batch size"
        },
    )
    from_scratch: Optional[bool] = field(
        default=True, metadata={"help": "Whether to pretrain from scratch"}
    )
    warmup_ratio: Optional[float] = field(
        default=0.1, metadata={"help": "Number of learning warmup steps to take"}
    )
    adam_beta1: Optional[float] = field(
        default=0.9, metadata={"help": "Parameter for the adam optimizer"}
    )
    adam_beta2: Optional[float] = field(
        default=0.95, metadata={"help": "Parameter for the adam optimizer"}
    )
    adam_epsilon: Optional[float] = field(
        default=1e-8, metadata={"help": "Parameter for the adam optimizer"}
    )
    weight_decay: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Parameter for the adam optimizer. Regularization to prevent weights from getting too big."
        },
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "LR scheduler type, such as cosine or linear."},
    )
    local_rank: Optional[int] = field(
        default=0, metadata={"help": "Used for multi-gpu"}
    )
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None, metadata={"help": "Path to deepspeed config if using deepspeed"}
    )
    peft: Optional[bool] = field(default=False, metadata={"help": "Whether to do PEFT"})


class HFLM_Local(HFLM):
    def get_model_info(self):
        return {}

def train_model():
    
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # If args.config is specified, use this script just to kick off a bunch
    # of jobs, and then exit from the script
    if script_args.config is not None:
        with open(script_args.config, "r") as file:
            config = SimpleNamespace(**yaml.safe_load(file))
        
        master_port = 29513
        for job in config.jobs:
            job = SimpleNamespace(**job)
            job_log = "job_log.txt"
            for seed in range(job.seeds):
                os.makedirs(os.path.join(config.raw_job_output_dir, job.name, str(seed)), exist_ok=True)
                command = f"bash train_llm_scheduler.sh \
'{os.path.join(config.raw_job_output_dir, job.name, str(seed), job_log)}' '{master_port}' \
'{os.path.join(config.raw_job_output_dir, job.name, str(seed))}' '{config.mmap_dataset_prefix}' \
'{seed}' '{job.sample_weights}' '{job.tokens}' '{job.per_device_train_batch_size}' '{job.model_id}' '{job.learning_rate}'"
                subprocess.call(command, shell=True)
                master_port += 1
        sys.exit()
    

    logger.info(f"Script parameters {script_args}")

    # set seed for reproducibility
    set_seed(script_args.seed)


    # load processed dataset
    prefix = script_args.mmap_prefix
    start_idx = np.load(prefix + "_start.npy")
    len_idx = np.load(prefix + "_len.npy")
    max_tokens = np.load(prefix + "_metadata.npy")

    if script_args.sample_wt_path in (None, "None"):
        prob_vector = len_idx / np.sum(len_idx)
        data_name = "default"
    else:
        prob_vector = np.load(script_args.sample_wt_path)
        assert((prob_vector >= 0).all())
        assert(np.allclose(sum(prob_vector),1))
        data_name, _ = os.path.splitext(os.path.basename(script_args.sample_wt_path))

    train_dataset = get_dataset_async(prob_vector=prob_vector, ctx_len=1024, memmaped_file=prefix + ".mmap", start_map=start_idx, len_map=len_idx, max_tokens=max_tokens, batch_size=10000)

    output_dir = f"{script_args.output_dir_prefix}/"

    token_count = script_args.token_count/world_size

    # define our hyperparameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        local_rank=script_args.local_rank,
        deepspeed=script_args.deepspeed,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=25,
        save_strategy="epoch",
        report_to="wandb",
        run_name=data_name,
        # push to hub parameters
        push_to_hub=False,
        # optimization parameters
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        learning_rate=script_args.learning_rate,
        seed=script_args.seed,
        max_steps=int(token_count/(1024*script_args.gradient_accumulation_steps*script_args.per_device_train_batch_size)),
        max_grad_norm=1.0,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        warmup_ratio=script_args.warmup_ratio,
        adam_beta1=script_args.adam_beta1,
        adam_beta2=script_args.adam_beta2,
        bf16=True,
        torch_compile=True,
        torch_compile_mode="max-autotune",
        adam_epsilon=script_args.adam_epsilon,
        weight_decay=script_args.weight_decay,
        lr_scheduler_type=script_args.lr_scheduler_type,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl"
    )

    # load trained model
    if script_args.from_scratch:
        hf_config = AutoConfig.from_pretrained(script_args.model_id)
        model = AutoModelForCausalLM.from_config(hf_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(script_args.model_id)

    if script_args.peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # load out data collator
    data_collator = DefaultDataCollator(return_tensors="pt")

    run = None
    if is_main_process:    
        np.save("prob_vector.npy", prob_vector)
        run = wandb.init(project="tthrush-pretrain-test", name=f"{script_args.output_dir_prefix}_{data_name}")
        run.log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".sh"))
        artifact = wandb.Artifact("prob_vector_artifact", type="dataset")
        artifact.add_file("prob_vector.npy")
        wandb.log_artifact(artifact)


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # train the model
    print("Training")
    trainer.train(script_args.resume_from_checkpoint)
    print("Trained")

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    if is_main_process:
        model.save_pretrained(output_dir+"/final/")
        tokenizer.save_pretrained(output_dir+"/final/")

    eval_lm = HFLM_Local(pretrained=model, tokenizer = tokenizer)

    return run, eval_lm



def eval_model(lm):
    results = lm_eval.simple_evaluate(
        model=lm,
        #model="hf",
        #model_args=f"pretrained={output_dir}/final/,trust_remote_code=True",
        tasks=["piqa","arc_easy","lambada_openai"],
        batch_size="auto",
        limit=5000,
        bootstrap_iters=1000,
        log_samples=True,
    )
    return results


def add_custom_metrics(results):
    for task_name, task_samples in results["samples"].items():
        # We have only verified this i/o for the following tasks
        if task_name not in ["piqa","arc_easy","sciq","lambada_standard","lambada_openai","lambada_openai_mt_de","lambada_openai_mt_fr","lambada_openai_mt_en","lambada_openai_mt_es","lambada_openai_mt_it"]:
            continue
        
        ll_correct_sum = 0
        num_ll_correct = 0
        ll_incorrect_sum = 0
        num_ll_incorrect = 0

        for sample in task_samples:
            if len(sample["resps"]) == 1:
                # If there is only one option (this is the case for lambada),
                # assume it is the correct option 
                ll_correct_sum += task_samples[0]["resps"][0][0][0]
                num_ll_correct += 1
            else:
                # If there are multiple options, then this task needs to have a target.
                for idx in range(len(sample["resps"])):
                    ll = sample["resps"][idx][0][0]
                    if idx == sample["target"]:
                        ll_correct_sum += ll
                        num_ll_correct += 1
                    else:
                        ll_incorrect_sum += ll
                        num_ll_incorrect += 1
        
        if num_ll_correct > 0:
            avg_ll_correct = ll_correct_sum/num_ll_correct
            results["results"][task_name]["avg_ll_correct"] = avg_ll_correct

        if num_ll_incorrect > 0:
            avg_ll_incorrect = ll_incorrect_sum/num_ll_incorrect
            results["results"][task_name]["avg_ll_incorrect"] = avg_ll_incorrect



if __name__ == "__main__":
    run, lm = train_model()
    if is_main_process:
        results = eval_model(lm)
        add_custom_metrics(results)
        del results["samples"]  # Samples were only needed for adding custom metrics, delete before printing
        print(results)
        for name, metrics in results['results'].items():
            run.summary[name] = metrics['acc,none']

            if "perplexity,none" in metrics:
                run.summary[name + "_perplexity,none"] = metrics['perplexity,none']

            if "avg_ll_correct" in metrics:
                run.summary[name + "_avg_ll_correct"] = metrics['avg_ll_correct']
            
            if "avg_ll_incorrect" in metrics:
                run.summary[name + "_avg_ll_incorrect"] = metrics['avg_ll_incorrect']

            if "avg_ll_correct" in metrics and "avg_ll_incorrect" in metrics:
                run.summary[name + "_avg_ll_delta"] = metrics['avg_ll_correct'] - metrics['avg_ll_incorrect']

        wandb.finish()
