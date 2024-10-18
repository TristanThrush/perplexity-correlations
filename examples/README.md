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
on a device with `nvcc`. Then you can switch to another device. Most of the requirements here are for the plethora of random Hugging Face models that you may want to specify, to get bits-per-byte (BPB) values from them.

Navigate to `get_eval_and_bpb_data` for an example of using a cluster
to get BPB values and evaluation scores from different groups of ~100
open-source language models on a sample RedPajama V2 and also a synthetic
pretraining dataset.

Navigate to `get_pretraining_sampling_dist` for an example of using
the information from `get_eval_and_bpb_data` to estimate a great training
sampling distribution and save reusable fastText training data filters for this
distribution. You should be able to take the fastText filters and easily plug them
into your LLM training pipeline, whatever it may be.

In the examples, we've tried to make it easy to swap in other pretraining
datasets, LLMs, evaluations, and cluster schedulers.
