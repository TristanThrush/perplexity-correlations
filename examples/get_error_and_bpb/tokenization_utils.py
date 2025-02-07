import torch
from transformers import AutoTokenizer
from typing import List, Tuple
from collections import defaultdict

def batch_tokenize_with_token_str_info(tokenizer, input_texts: List[str], truncation=False) -> Tuple:
    encoded_batch = tokenizer(
        input_texts, padding=True, truncation=truncation, return_tensors="pt"
    )
    token_strings = [[t.replace("Ġ", "").strip() for t in tokenizer.convert_ids_to_tokens(input_ids)] for input_ids in encoded_batch.input_ids]
    return encoded_batch, token_strings


def batch_tokenize_with_char_step(
    tokenizer, input_texts: List[str], step_size: int, truncation=False
) -> Tuple:
    """
    Tokenizes a batch of input strings and maps character step positions to the closest token indices.

    Args:
        tokenizer: Hugging Face tokenizer object.
        input_texts (List[str]): A list of strings to tokenize.
        step_size (int): The step size in characters.
        truncation (bool): Whether to truncate inputs at the model's max length.

    Returns:
        Tuple containing:
        - Encoded batch output from tokenizer
        - A tensor of token indices marking step start (inclusive) and end (exclusive) positions
        - A list of character step indices for each input
    """
    # Generate character step positions for each text
    char_steps_list = [
        list(range(0, len(text), step_size)) + ([len(text)] if len(text) % step_size != 0 else [])
        for text in input_texts
    ]
    
    # Tokenize while preserving offset mappings
    encoded_batch = tokenizer(
        input_texts, return_offsets_mapping=True, padding=True, truncation=truncation, return_tensors="pt"
    )
    
    offset_mappings = encoded_batch["offset_mapping"].tolist()  # Convert tensor to list
    
    step_token_indices = []
    for i, (text, char_steps) in enumerate(zip(input_texts, char_steps_list)):
        offsets = offset_mappings[i]  # List of (start, end) tuples

        token_indices = []
        for char_index in char_steps:
            for j, (start, end) in enumerate(offsets):
                if start <= char_index < end:
                    token_indices.append(j)
                    break
            else:
                token_indices.append(len(offsets) - 1)  # Fallback to last token if no match
        
        # Ensure the last token index is properly included as an exclusive endpoint
        if token_indices[-1] != len(offsets):
            token_indices.append(len(offsets))
        
        # Form pairs of (start_token_index, end_token_index) ensuring end is exclusive
        step_pairs = [
            (token_indices[k], token_indices[k + 1])
            for k in range(len(token_indices) - 1)
        ]
        step_token_indices.append(step_pairs)
    
    del encoded_batch["offset_mapping"]
    return encoded_batch, step_token_indices, char_steps_list


def batch_tokenize_with_percentage_based_indices(
    tokenizer, input_texts: List[str], percentage_positions_list: List[float], truncation=False
) -> List[Tuple[List[str], List[int]]]:
    """
    Tokenizes a batch of input strings and maps percentage positions to the closest token indices.

    Args:
        tokenizer: Hugging Face tokenizer object.
        input_texts (List[str]): A list of strings to tokenize.
        percentage_positions_list (List[List[float]]): A list of lists containing percentage positions (0 to 1) 
                                                       for each string.
        truncation (bool): Whether to truncate inputs at the model's max length.

    Returns:
        List[Tuple[List[str], List[int]]]: A list of tuples where each tuple contains:
            - The list of tokens for the string
            - A list of nearest token indices corresponding to the input percentage positions
    """

    # Convert percentage positions into character indices
    char_indices_list = [
        [max(0, min(len(text) - 1, int(p * len(text)))) for p in percentage_positions_list]  # Convert percentages to char indices
        for text in input_texts
    ]

    # Tokenize using return_tensors="pt" for PyTorch
    encoded_batch = tokenizer(input_texts, return_offsets_mapping=True, padding=True, truncation=truncation, return_tensors="pt")

    # Convert tensor values to lists for processing
    offset_mappings = encoded_batch["offset_mapping"].tolist()  # Convert tensor to list

    suffix_indices = []
    for i, (text, char_indices) in enumerate(zip(input_texts, char_indices_list)):
        offsets = offset_mappings[i]  # Now it's a list of (start, end) tuples

        nearest_token_indices = []
        for char_index in char_indices:
            nearest_token_index = None
            min_distance = float("inf")

            for j, (start, end) in enumerate(offsets):
                if start <= char_index < end:  # Character is within this token
                    nearest_token_index = j
                    break
                # If not directly within a token, find the closest one
                distance = min(abs(char_index - start), abs(char_index - end))
                if distance < min_distance:
                    min_distance = distance
                    nearest_token_index = j
            
            if nearest_token_index is None:
                # failed
                return encoded_batch, None, char_indices_list
            nearest_token_indices.append(nearest_token_index)

        suffix_indices.append(nearest_token_indices)
    
    del encoded_batch["offset_mapping"]
    return encoded_batch, torch.tensor(suffix_indices), char_indices_list


def compute_average_loss_from_index_tuples(loss, token_index_tuples, attention_mask):

    """
    Computes the average loss starting from each token index in a batch.

    Args:
        loss (torch.Tensor): Per-token loss tensor of shape (batch_size, seq_length-1).
        token_index_tuples (list): Tensor of shape (batch_size, num_indices), containing start token indices.
        attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_length).

    Returns:
        list: A list of lists containing the average loss for each specified token index.
    """
    batch_size, seq_len_minus_1 = loss.shape

    # Extract the valid attention mask for shift_labels (excluding the first token)
    valid_attention_mask = attention_mask[..., 1:]  # Shape: (batch_size, seq_length-1)

    # Adjust token indices to match the shifted loss indices
    shifted_token_index_tuples = []
    for tuples in token_index_tuples:
        shifted_tuples = []
        for t in tuples:
            shifted_tuples.append((max(t[0]-1,0), max(t[1]-1,0)))
        shifted_token_index_tuples.append(shifted_tuples)

    # Create a list to store the averaged losses
    avg_losses_list = []

    for index, tuples in enumerate(shifted_token_index_tuples):
        avg_losses = []
        for t in tuples:

            if t[0] >= seq_len_minus_1:  # If indices are out of range, assign NaN
                avg_losses.append(float('nan'))
                continue

            # Compute sum of loss starting from `start_idx`
            remaining_loss = loss[index, t[0]:t[1]].sum()

            # Compute the number of valid tokens from `start_idx`
            num_valid_tokens = valid_attention_mask[index, t[0]:t[1]].sum().float()

            # Compute average loss from `start_idx`, avoiding division by zero
            avg_losses.append(remaining_loss / num_valid_tokens if num_valid_tokens > 0 else float('nan'))
        avg_losses_list.append(avg_losses)

    return avg_losses_list



def compute_average_loss_from_indices(loss, token_indices, attention_mask):
    """
    Computes the average loss starting from each token index in a batch.

    Args:
        loss (torch.Tensor): Per-token loss tensor of shape (batch_size, seq_length-1).
        token_indices (torch.Tensor): Tensor of shape (batch_size, num_indices), containing start token indices.
        attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_length).

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_indices) containing the average loss 
                      for each specified token index.
    """
    batch_size, seq_len_minus_1 = loss.shape
    num_indices = token_indices.shape[1]  # Fixed number of indices per batch item

    # Extract the valid attention mask for shift_labels (excluding the first token)
    valid_attention_mask = attention_mask[..., 1:]  # Shape: (batch_size, seq_length-1)

    # Adjust token indices to match the shifted loss indices
    token_indices = token_indices - 1  # Since loss is shifted by 1 position

    # Ensure token indices are within valid range
    token_indices = torch.clamp(token_indices, min=0)

    # Create a tensor to store the averaged losses
    avg_losses = torch.zeros((batch_size, num_indices), device=loss.device)

    for i in range(batch_size):
        for j in range(num_indices):
            start_idx = token_indices[i, j]
            if start_idx >= seq_len_minus_1:  # If index is out of range, assign NaN
                avg_losses[i, j] = float('nan')
                continue

            # Compute sum of loss starting from `start_idx`
            remaining_loss = loss[i, start_idx:].sum()

            # Compute the number of valid tokens from `start_idx`
            num_valid_tokens = valid_attention_mask[i, start_idx:].sum().float()

            # Compute average loss from `start_idx`, avoiding division by zero
            avg_losses[i, j] = remaining_loss / num_valid_tokens if num_valid_tokens > 0 else float('nan')

    return avg_losses  # Shape: (batch_size, num_indices)


def compute_token_loss_dicts(loss, attention_mask, token_strings):
    batch_size, seq_len_minus_one = loss.shape
    seq_len = seq_len_minus_one + 1
    
    assert attention_mask.shape == (batch_size, seq_len)
    assert len(token_strings) == batch_size
    assert all(len(ts) == seq_len for ts in token_strings)
    
    valid_attention_mask = attention_mask[..., 1:]
    
    shifted_token_strings = []
    for strings in token_strings:
        shifted_strings = strings[1:]
        shifted_token_strings.append(shifted_strings)
    
    token_loss_dicts = []
    
    for batch_idx in range(batch_size):
        token_loss_dict = defaultdict(lambda: [0.0, 0])  # {token: [total_loss, count]}
        
        for token_idx in range(seq_len_minus_one):  # loss is seq_len-1
            if valid_attention_mask[batch_idx, token_idx] == 1:  # Shifted alignment
                token = shifted_token_strings[batch_idx][token_idx]  # Shifted tokens
                token_loss_dict[token][0] += loss[batch_idx, token_idx].item()
                token_loss_dict[token][1] += 1
        
        # Compute average loss per token
        for token in token_loss_dict:
            total_loss, count = token_loss_dict[token]
            token_loss_dict[token] = (total_loss / count if count > 0 else 0.0, count)
        
        token_loss_dicts.append(token_loss_dict)
    
    return token_loss_dicts


# Example Usage
if __name__ == "__main__":
    # Load some tokenizers
    tokenizers = []
    tokenizers.append(AutoTokenizer.from_pretrained("facebook/opt-2.7b"))
    tokenizers.append(AutoTokenizer.from_pretrained("gpt2"))
    tokenizers.append(AutoTokenizer.from_pretrained("EleutherAI/pythia-160m"))
    

    # Input batch of texts and corresponding lists of percentage positions (0 to 1)
    input_texts = [
        "The quick brown fox jumps over the lazy dog. Then the lazy dog went and trafed some stocks this is jyst noise104594yhh at this point </html-trash> hehe yeah.",
        "0239rh4 RHwhat are you talking about? https://huggingface.co is the the url. Hugging Face is an AI entrepreneurship company. Yeahhhhhhh right entrepernuership (I intentionally spelled that word wrong btw. I am just rampling because I am part of code thats testing a string that needs to be a certain length."
    ]
    percentage_positions_list = [0.10, 0.50, 0.90]  # Percentages to locate positions

    # Process batch with PyTorch tensors
    print("showcasing batch_tokenize_with_percentage_based_indices")
    for tokenizer in tokenizers:
        tokenizer.pad_token = tokenizer.eos_token
        print("tokenizer:", tokenizer)
        inputs, suffix_indices, char_indices_list = batch_tokenize_with_percentage_based_indices(tokenizer, input_texts, percentage_positions_list)
        print(inputs)
        print(suffix_indices)
        print(char_indices_list)
        for ids in inputs["input_ids"].tolist():
            tokens = tokenizer.convert_ids_to_tokens(ids)
            print(tokens)
        print()
        '''
        # Print results
        for i, (tokens, token_indices) in enumerate(batch_results):
            print(f"\nText {i+1}: {input_texts[i]}")
            print(f"Tokens: {tokens}")
            print(f"Nearest Token Indices: {token_indices}")
            print(f"Nearest Tokens: {[tokens[idx] for idx in token_indices]}")
        '''

    print("showcasing batch_tokenize_with_char_step")
    
    for tokenizer in tokenizers:
        tokenizer.pad_token = tokenizer.eos_token
        print("tokenizer:", tokenizer)
        inputs, step_token_indices, char_indices_list = batch_tokenize_with_char_step(tokenizer, input_texts, 10)
        print(inputs)
        print(step_token_indices)
        print(char_indices_list)
        index = 0
        for ids in inputs["input_ids"].tolist():
            print("len step token indices:", str(len(step_token_indices[index])))
            tokens = tokenizer.convert_ids_to_tokens(ids)
            print(tokens)
            for t in step_token_indices[index]:
                print(tokens[t[0]:t[1]])
            index += 1
            print()
        print()

    for tokenizer in tokenizers:
        tokenizer.pad_token = tokenizer.eos_token
        inputs, token_strings = batch_tokenize_with_token_str_info(tokenizer, input_texts)
        print(token_strings)
