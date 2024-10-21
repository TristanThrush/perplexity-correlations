# Examples

This directory contains examples of code that might be useful if you
want to use our package in practice.

For these examples to work, get a machine with CUDA (hopefully you have a cluster
and you can just grab something like an A100 machine for a few min), and then run:

```
pip install -r requirements.txt
```

Note that you don't need to run all of the scripts in these examples with a CUDA machine,
but some of the pip packages (like `flash-attn`) will fail to install unless you are installing
on a device with `nvcc`. Then you can switch to another device. Most of the requirements here are for the plethora of random Hugging Face models from which you might want to get bits-per-byte (BPB) values.

Navigate to `get_error_and_bpb/` for examples of getting BPB values and evaluation scores.
The examples compute BPB from different groups of ~100 open-source language models on a sample RedPajama V2
and also a synthetic pretraining dataset. They also compute evals on many different benchmarks using the
Eleuther Eval Harness.

Navigate to `get_fasttext_filter/` for examples of using
the information from `get_error_and_bpb/` to estimate good pretraining
sampling distributions. The examples save reusable fastText training data filters,
which you can plug into your LLM training pipeline, whatever it may be.

In the examples, we've tried to make it easy to swap in other pretraining
datasets, LLMs, evaluations, and cluster schedulers.
