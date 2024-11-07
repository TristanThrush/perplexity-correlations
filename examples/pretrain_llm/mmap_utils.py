from transformers import AutoTokenizer
import torch
import numpy as np
from datasets import IterableDataset
from tqdm import tqdm
import concurrent.futures

def tokenize_and_mmap(seq_id_iterator: list[(str, list[int])], tokenizer: AutoTokenizer, max_tokens:int, ctx_len: int, file_prefix: str) -> (np.array, np.array, np.array, list[str]):
    # given a list of string to tokenize, tokenize each one and write to a memmap file in order
    # return the memmap array and the starting index and the length of each piece of text
    tokenized_mmap_file = np.memmap(file_prefix+'.mmap', dtype='int32', mode='w+', shape=(max_tokens))
    len_list = []
    cur_idx = 0
    id_selected = []
    for id, tok_list in tqdm(seq_id_iterator):
        tokens = np.array(tok_list)
        total_tokens = tokens.size
        if cur_idx + total_tokens > max_tokens:
            # we could add this last truncated bit, but forget it - messes up indexing.
            #truncated_token_ct = max_tokens - cur_idx
            #tokenized_mmap_file[cur_idx:] = tokens[:truncated_token_ct]
            #len_list.append(truncated_token_ct)
            break
        if total_tokens >= ctx_len:
            tokenized_mmap_file[cur_idx:cur_idx+total_tokens] = tokens
            cur_idx += total_tokens
            len_list.append(tokens.size)
            id_selected.append(id)
        if len(len_list) % 100000 == 0:
            # periodically flush writes to disk, clear memory
            tokenized_mmap_file.flush()
            tokenized_mmap_file = np.memmap(file_prefix+'.mmap', dtype='int32', mode='r+', shape=(max_tokens))
    start_index = np.array([0] + np.cumsum(len_list)[:-1].tolist())
    len_array = np.array(len_list)
    # dump both arrays
    np.save(file_prefix+'_start.npy', start_index)
    np.save(file_prefix+'_len.npy', len_array)
    np.save(file_prefix+'_metadata.npy', np.array(max_tokens))
    np.save(file_prefix+'_id.npy', np.array(id_selected))
    return tokenized_mmap_file, start_index, len_array, id_selected

def sample_from_vec(prob_vector: np.array, batch_size: int, ctx_len: int, memmapped_array: np.array, start_map: np.array, len_map: np.array, gen: np.random.Generator = np.random.Generator(np.random.PCG64())):
    # samples tokens in a weighted way from documents.
    # samples a doc proportionally to prob_vector.
    # within each doc, sample a window of ctx_len uniformly at random.
    # returns the sampled batch of token indices
    assert(np.min(len_map) >= ctx_len)  # can kill this if slow..
    # get the document ids
    #doc_ids = np.array(random.choices(range(len(prob_vector)), weights=prob_vector, k=batch_size)) #random.choices is slightly faster than numpy
    doc_ids = gen.choice(len(prob_vector), p=prob_vector, size=batch_size)
    # now get the offsets -
    offset_ids = np.random.randint(len_map[doc_ids] - ctx_len + 1)
    start_points = start_map[doc_ids] + offset_ids
    # do some fancy reshaping to do vectorized indexing
    flattened_idx = np.add.outer(start_points, np.arange(ctx_len)).reshape(ctx_len*batch_size)
    sampled_batch = memmapped_array[flattened_idx].reshape(batch_size, ctx_len)
    return torch.LongTensor(sampled_batch), torch.ones(sampled_batch.shape)

def get_dataset(prob_vector:np.array, ctx_len: int, memmaped_file: str, start_map: np.array, len_map: np.array, max_tokens: int, batch_size = 10000):
    def gen():
        rng = np.random.Generator(np.random.PCG64())
        while True:
            temp_memmap = np.memmap(memmaped_file, dtype='int32', mode='r', shape=(max_tokens))  # reinitialize memmap for memory
            sampled_batches, masks = sample_from_vec(prob_vector, batch_size, ctx_len, temp_memmap, start_map, len_map, rng)
            for i in range(batch_size):
                yield {
                    "input_ids": sampled_batches[i,:].squeeze(),
                    "labels": sampled_batches[i,:].squeeze(),
                    "attention_mask": masks[i,:].squeeze()
                }
    print('get_dataset')
    return IterableDataset.from_generator(gen)

import time
def get_dataset_async(prob_vector: np.array, ctx_len: int, memmaped_file: str, start_map: np.array, len_map: np.array,
                      max_tokens: int, batch_size = 10000):
    # async version of the above - used to overlap reads and GPU computation
    def gen():
        rng = np.random.Generator(np.random.PCG64())
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_batch = executor.submit(sample_from_vec, prob_vector, batch_size, ctx_len,
                                           np.memmap(memmaped_file, dtype='int32', mode='r', shape=(max_tokens)),
                                           start_map, len_map, rng)

            while True:
                start = time.time()
                # Wait for the future to complete and get the result
                sampled_batches, masks = future_batch.result()

                # Submit the next batch generation
                future_batch = executor.submit(sample_from_vec, prob_vector, batch_size, ctx_len,
                                               np.memmap(memmaped_file, dtype='int32', mode='r', shape=(max_tokens)),
                                               start_map, len_map, rng)

                end = time.time()
                print('batch overhead '+str(end-start)+'(s)')
                for i in range(batch_size):
                    yield {
                        "input_ids": sampled_batches[i,:].squeeze(),
                        "labels": sampled_batches[i,:].squeeze(),
                        "attention_mask": masks[i,:].squeeze()
                    }

    print('get_dataset')
    return IterableDataset.from_generator(gen)

# Plan for without replacement sampler:
# Do modulo rank
# Do modulo seq len
# Convert prob vector into token counts
# Store a dict of the remaining token counts to sample for each page
# Once the remaining token counts gets to zero, remove it from the dict
# If the remaining token count dict is empty, start again. This means we are doing more than one epoch.
# If the remaining token counts is smaller than seq len, keep sampling from other pages, adding an eos token inbetween, until you are at seq len.



if __name__ == '__main__':
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('Tokenizer fastness:'+str(tokenizer.is_fast))
    max_tokens = 1024
    test_strings = ["the quick brown fox jumps over the lazy dog", "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."]
    print(tokenizer(test_strings[0], return_tensors="pt"))
    tokenized_list = [tokenizer(seq+tokenizer.eos_token, return_tensors="pt")['input_ids'][0] for seq in test_strings]
    merged_seq, start_map, len_map, id_list = tokenize_and_mmap(enumerate(tokenized_list), tokenizer, max_tokens,4, 'test')
    dataset = get_dataset_async(np.array([0.1, 0.9])[id_list], 4, 'test.mmap', start_map, len_map, max_tokens)
    for i, data in enumerate(dataset):
        print(tokenizer.decode(data['input_ids'].numpy().tolist()))
        #_ = tokenizer.decode(data[0])
        if i > 100:
            break
