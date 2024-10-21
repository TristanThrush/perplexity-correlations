# Compute Estimate, Train and Save a fastText Filter

You can run the `get_fasttext_filter.py` script in this directory to compute perplexity
correlation estimators, using the BPB and error `.csv` files in the `get_error_and_bpb/` directory.
Also note that you need to have the required chunked dataset in the `get_error_and_bpb/` directory too.
The script saves fastText pretraining data filters targeting specified benchmarks.

Check out some of the example config `.yml` files in this directory to see some of the options that
can be specified. While many options should hopefully be self-explanatory, it is worth noting a few
of them:

* `hf_tokenizer_name` is expected to be the tokenizer that you will actually use for pretraining, but realistically you could use many standard tokenizers. The goal is to use this tokenizer to get the relative token sizes of the domains/ids/chunks that you are going to be using for pretraining. This only needs to be approximate. We just want our algorithm to know the relative token counts of each domain/id/chunk so that it doesn't suggest sampling huge weights for an e.g. low-token domain (this would result in data duplication).
* `desired_filter_ratio` is expected to be the filtering ratio that you will actually use for pretraining. For example, if you have a pretraining dataset and your goal is to use perplexity correlations to filter away 90% of it (so only 10% is remaining), then you would set desired_filter_ratio to 0.1.
* `target_benchmark_groups`. A unique fastText pretraining filter will be trained using perplexity correlations and saved for each target benchmark group. Results for all configs are saved in directories called `fasttext_models`, `fasttext_info`, `fasttext_datasets`, and are saved based on the name of the target benchmark group. So the benchmark group name should be unique even accross different configs. Benchmark groups tell our algorithm which benchmark to target - if multiple benchmkarks are specified then the average accross those benchmarks is used.

Now, what exactly is in the output directories `fasttext_models`, `fasttext_info`, `fasttext_datasets`?

* `fasttext_info` contains a little mini-eval for every fastText model trained. We've uploaded the files generated in `fasttext_info` after running `get_fasttext_filter.py` for every example config `.yml`, so you can gain an intuition about what our approach selects. It shows the language distribution for the sample of the pretraining dataset that you used before and after running our estimate and after running fastText. So, if the benchmark is in Italian, it would be encouraging to see that our fastText model is picking out much more Italian than a uniform random sample would. `fasttext_info` also includes the top 10 domains/ids/chunks that are selected by the estimate and by the projected estimate. Again, if our benchmark is e.g. Italian, it would be encouraging to see the top 10 domains be Italian.
* `fasttext_models` will contain the fastText models generated after running our script that you can then take and use with your pretraining filtering script. We didn't upload them here due to size.
* `fasttext_datasets` will contain the datasets that were generated from our perplexity correlations extimate to train and eval the fastText models. Again, we didn't upload them here due to size.
