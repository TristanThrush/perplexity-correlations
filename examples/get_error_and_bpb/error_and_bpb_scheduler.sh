# Note that the nlprun script currently assumes that you have a conda environment
# called `perplexity-correlations` that has all of the dependencies in requirements.txt
# here.
#
# If you want to change this script to work with a different scheduler, it should
# hopefully be easy. Just make sure it calls the get_error_and_bpb.py with the
# exact same arguments as below. Notice the `resume` flag at the end. For any reason
# if the jobs fail, then this flag says to not redo the cached bpb values and start
# computing bpb where the script left off.

JOBID1=$(nlprun -q jag -r 60G -g 1 -c 16 -a perplexity-correlations -o \
    $1/job_log.txt "python get_error_and_bpb.py \
        --raw_job_output_path $1 \
        --hf_llm_family $2 \
        --hf_llm_name $3 \
        --eleuther_eval_names $4 \
        --eleuther_eval_metrics $5 \
        --eleuther_eval_lower_is_better $6 \
        --chunked_pretraining_data_sample $7 \
        --resume \
        --half_precision \
        --save_model_info \
        " | grep -oP "Submitted batch job \K[0-9]+")

