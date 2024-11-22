import argparse
import fasttext
import uuid
import random
from datasets import load_dataset

# Function to classify a batch of examples
def classify_batch(examples, model_path):
    """Classify a batch of examples using the FastText model and add unique IDs and probabilities."""
    # Load the model inside the function to ensure it's loaded separately for each worker
    model = fasttext.load_model(model_path)
    texts = [text.replace("\n", " ").strip() for text in examples.get("text", [])]  # Sanitize text
    labels, probs = model.predict(texts)

    probabilities = []
    for label, prob in zip(labels, probs):
        # Check for __label__hq or __label__include and get the probability
        if '__label__hq' in label or '__label__include' in label:
            probabilities.append(prob.item())
        else:
            probabilities.append(1-prob.item())

    examples["prob_hq"] = probabilities
    return examples

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Classify dataset examples and filter by probability.")
    parser.add_argument("--model_path", type=str, help="Path to the FastText model file.")
    parser.add_argument("--filter_percentage", type=float, required=True, 
                        help="Percentage of examples to filter out.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Set the random seed
    random.seed(args.seed)

    # Load the Hugging Face dataset
    dataset = load_dataset("dclm_1b_1x_pool_hf", num_proc=208)["train"] # Replace with your dataset

    if args.model_path:
        classified_dataset = dataset.map(
            lambda examples: classify_batch(examples, args.model_path),
            batched=True,
            batch_size=100000,
            num_proc=208,
        )

        # Calculate the threshold for filtering
        total_examples = len(classified_dataset)
        filter_count = int(total_examples * args.filter_percentage / 100)

        # Sort the dataset by prob_hq and get the threshold value
        sorted_dataset = classified_dataset.sort("prob_hq", reverse=True)
        threshold_value = sorted_dataset[filter_count]["prob_hq"]

        # Filter out the lowest percentage of prob_hq values
        filtered_dataset = sorted_dataset.select(range(total_examples - filter_count))
        for i in range(10):
            print(sorted_dataset[i]["url"])
            print(sorted_dataset[i]["text"]) 
            print(sorted_dataset[i]["prob_hq"])
            print()
    else:
        # Randomly filter away the specified percentage of the dataset
        total_examples = len(dataset)
        filter_count = int(total_examples * args.filter_percentage / 100)
        indices_to_keep = set(random.sample(range(total_examples), total_examples - filter_count))

        filtered_dataset = dataset.select(indices=list(indices_to_keep))

    # Retain only the text, url, and id columns
    final_dataset = filtered_dataset.remove_columns(
        [col for col in filtered_dataset.column_names if col not in {"text", "url", "prob_hq"}]
    )

    final_dataset = final_dataset.map(lambda example: {"id": example.get("id", str(uuid.uuid4()))}, num_proc=208)

    # Save the filtered dataset
    save_path = f"{args.model_path}_filtered_dclm_pool_hf" if args.model_path else "random_filtered_dclm_pool_hf"
    final_dataset.save_to_disk(save_path, num_proc=208)

if __name__ == "__main__":
    main()
