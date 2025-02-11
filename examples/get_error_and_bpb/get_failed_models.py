import yaml
import pandas as pd
import argparse
from types import SimpleNamespace

parser = argparse.ArgumentParser()

parser.add_argument("--config")

args = parser.parse_args()

with open(args.config, "r") as file:
    config = SimpleNamespace(**yaml.safe_load(file))

#for columns in [pd.read_csv(config.error_output_csv).columns, pd.read_csv(config.bpb_output_csv_prefix + "_domain.csv")]:
for columns in [pd.read_csv(config.bpb_output_csv_prefix + "_domain.csv")]:
    print("failures:")
    for llm in config.llms:
        for name in llm["hf_names"]:
            if str((llm["family"], name)) not in columns:
                print((llm["family"], name))
    print()

