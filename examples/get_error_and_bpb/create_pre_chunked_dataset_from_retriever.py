import yaml
from types import SimpleNamespace
import argparse
from index_wikipedia import query_faiss_index, load_faiss_index
from datasets import Dataset, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = SimpleNamespace(**yaml.safe_load(file))

index, titles, texts, urls = load_faiss_index()

documents_dict = {}
def build_documents_dict(example):
    query = " ".join([example[column] for column in config.benchmark_columns])
    results = query_faiss_index(index, titles, texts, urls, query, config.k)
    for result in results:
        documents_dict[result["url"]] = {"title": result["title"], "text": result["text"], "count": documents_dict.get(result["url"], {"count": 0})["count"] + 1}

ds = load_dataset(config.benchmark_dataset, config.benchmark_dataset_subset, split=config.benchmark_dataset_split)
ds.map(build_documents_dict)

documents_dict_for_dataset = {"title": [], "text": [], "count": [], "url": [], "domain": []}
for key, value in documents_dict.items():
    documents_dict_for_dataset["title"].append(value["title"])
    documents_dict_for_dataset["text"].append(value["text"])
    documents_dict_for_dataset["count"].append(value["count"])
    documents_dict_for_dataset["url"].append(key)
    documents_dict_for_dataset["domain"].append("wikipedia")

document_ds = Dataset.from_dict(documents_dict_for_dataset)
document_ds = document_ds.sort("count", reverse=True)
document_ds = document_ds.select(range(config.global_document_limit))
document_ds.save_to_disk(config.output_path)

