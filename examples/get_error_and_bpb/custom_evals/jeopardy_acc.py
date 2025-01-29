from datasets import load_dataset
import torch

def jeopardy_accuracy(model, tokenizer, device):

    def calculate_jeopardy_accuracy(question, answer):
        # Concatenate the question as the model's input
        input_text = question
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Calculate the number of tokens in the answer
        answer_token_count = len(tokenizer(answer, return_tensors="pt")["input_ids"][0])

        with torch.no_grad():
            # Generate output limited to the length of the answer
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=answer_token_count
            )
        
        # Decode the generated output
        filtered_outputs = [
            token for token in outputs[0]
            if token not in tokenizer.all_special_ids
        ]
        generated_answer = tokenizer.decode(filtered_outputs[-answer_token_count:])
        
        # Compare the generated answer with the correct answer (case insensitive)
        return generated_answer.strip().lower() == answer.strip().lower()

    ds = load_dataset("csv", data_files="custom_evals/jeopardy.csv")["train"]
    ds = ds.shuffle(seed=42)
    fs_prompt_ds = ds.select(range(10))
    fs_prompt = ""
    for example in fs_prompt_ds:
        fs_prompt += "|Question:" + example[" Question"] + "|Answer:" + example[" Answer"]
    
    sampled_ds = ds.select(range(10, 5010))
    
    # Calculate accuracy
    correct_count = 0
    total_count = 0
    
    for example in sampled_ds:
        question = fs_prompt + "|Question:" + example[" Question"] + "|Answer:"
        answer = example[" Answer"]
        if calculate_jeopardy_accuracy(question, answer):
            correct_count += 1
        total_count += 1

    accuracy = correct_count / total_count
    return accuracy
