# Getting BPB Matrix and Error Vectors

This directory has two python scripts:

* `chunk_pretraining_data_sample.py`, which allows you to specify a training data sample that it then chunks into pieces that fit into the contexts of LLMs that you care about.
* `get_error_and_bpb.py`, which allows you to specify LLMs, benchmarks, and a chunked dataset. This information is used to extract bits-per-byte values and benchmark errors that feed into our perplexity-correlations package.

## Chunk Pretraining Data

The first step is deciding what data you want to get a bunch of BPB
values on. Ideally, this should be an i.i.d. sample from a dataset that you want to pretrain with.
We need to chunk this data sample into pieces that will fit into the context of
a bunch of LLMs. We do this by using a script that uses some arbitrary reference tokenizer
(like the one from Llama 2) and a token threshold (like 256), and then creating a version
of the pretraining data sample that is split into these chunks. Check out `chunker_configs`
for a few examples of what can be specified. You should be able to use practically any Hugging Face text
dataset as long as it has a text column and id column. You can then run `chunk_pretraining_data_sample.py`
like this:

`python chunk_pretraining_data_sample.py --config chunker_configs/rpjv2_sample_chunker_config.yml`

This script will save the result as a Hugging Face dataset in a new directory called `chunked_datasets`.

It is reccomended that you get a machine with lots of CPUs and memory for this. You can set the
number of CPUs in the config. If you get unusual errors, it is probably because an example in a
pretraining dataset has a lot of text and required too much memory to load before chunking.

## Get Errors and BPB

The next step is to actually compute the benchmark error vectors and the BPB matrix that we
need for `perplexity-correlations` to work. We do this with the `get_error_and_bpb.py` script, which
takes in a config that you can use to specify the path to the chunked dataset
from the previous step, along with and evaluations available via the
[Eleuther Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness), and causal LLMs
available on Hugging Face. You can check out examples of what can be speficied in
`error_and_bpb_configs`.

The python script spins up a separate job for each LLM to get BPB values on the
chunked pretraining data sample, and benchmark errors on the evals. The jobs are spun up via the
command in `error_and_bpb_scheduler.sh` which is a short file that uses the Stanford Cluster's `nlprun`
scheduler - this is `slurm` under the hood. It should be easy to change
this file to be in the syntax of whatever scheduler you use on your machine. The 
command in `error_and_bpb_scheduler.sh` calls the python script itself. The way this works
is that the python script, when given a config `.yml` as an argument, uses
`error_and_bpb_scheduler.sh` to spin up jobs that call the python script, but this time
with arguments specifying a specific job for a specific LLM, and without `config.yml`
as an argument.

Assuming you modify `error_and_bpb_scheduler.sh` to work for your machine, you should be
able to run:

```
python get_error_and_bpb.py --config error_and_bpb_configs/basic_test_config.yml
```

and then check with `squeue` (or an equivalent) to see that the jobs started.
In the case of the above example, it is just a little test and only one job will start.
For every LLM job that completes, you will see rows and columns
asyncronously added to the specified error `.csv` file and the specified BPB `.csv` files.
The error BPB will be #LLMs x #benmarks. There will be multiple output BPB csv files: they will
be #LLMs x #chunks, #LLMs x #IDs, (and #LLMs x #Domains if you specified a domain column
in your dataset). They have BPB values at different levels of granularity depending on whether you
want to run our estimate at the domain level, page (id) level, or chunk level. Note that you can't
simply get e.g. the domain-level matrix by averaging over the page-level matrix: we need to use a
weighted average of losses based on how many tokens are in a page - the script handles this
under the hood for you.

`get_error_and_bpb.py` also automatically removes chunks for all LLMs in the computation of the domain and
ID BPB `.csv` files, if an LLM got a NaN loss in that chunk. This typically results in a few domain and ID
removals, but if you have a hugely problematic LLM, then you could get final BPB csvs that have almost no
entries! Luckily, information from the LLM jobs is cached elsewhere and so you can just delete the BPB
`.csv`s, remove that bad LLM and run `get_error_and_bpb.py` again - it should complete quickly for all the
good LLMs that were already run and it should save new BPB csvs that are much more populated. If you don't
want to redo evaluations in your config, you can just make it an empty list. Similarly, the script will remove a benchmark row in the error `.csv` if even one LLM job does not include that benchmark - so do
not alter the benchmarks in a config and re-run the script with that config unless you change the save location of the error `.csv`.

The cache that each LLM job creates can be found in a directory that the script creates, which is specified
in the example configs. There will be subdirectories for each LLM. Right now, `error_and_bob_scheduler.sh`
saves a job log for each model in `job_log.txt` - we reccomend that you implement this behavior with your
scheduler too. You can monitor progress or errors by checking this log. Common issues may include:
* The chunks are too big for a particular LLM's context. Youll either have to re-chunk the dataset or remove that LLM.
* A model download (particularly for Meta) may require that you are logged into your Hugging Face account via huggingface-cli and that you have already accepted the terms of use by going to that model's Hugging Face page.
* A model may run out of GPU memory. In `get_error_and_bpb.py` you can lower the default number for `--hf_llm_batch_size` and then try again.

## Existing job outputs for you to play with

We've run the two scripts in this directory already, for all example configs!
Due to size, we don't include the chunked datasets, or the cache files for each LLM job.
But we have still uploaded the error and BPB `.csv` files in `error_csvs` and `bpb_csvs`.
Note that `bpb_csvs/chunked_rpjv2_sample_bpb_domain.csv` and `error_csvs/error.csv` are the data
that were used to train the estimators in the Perplexity Correlations paper, and result from running:

```
python chunk_pretraining_data_sample.py --config chunker_configs/rpjv2_sample_chunker_config.yml
python get_error_and_bpb.py --config error_and_bpb_configs/error_and_rpjv2_bpb_config.yml
```

(tecnhically, we reimplemented the code from the paper, but have confirmed that this data results in
fastText classifiers with essentially the same behavior).
