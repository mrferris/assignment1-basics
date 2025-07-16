import os
from typing import BinaryIO
import regex as re
from collections import Counter
from multiprocessing import cpu_count, Pool

PRE_TOKENIZATION_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PRE_TOKENIZATION_REGEX = re.compile(PRE_TOKENIZATION_REGEX)

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # ## Usage
    # with open(..., "rb") as f:
    #     boundaries = find_chunk_boundaries(
    #         f, num_processes, "<|endoftext|>".encode("utf-8"))
        
    #     # The following is a serial implementation, but you can parallelize this 
    #     # by sending each start/end pair to a set of processes.
    #     for start, end in zip(boundaries[:-1], boundaries[1:]):
    #         f.seek(start)
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #         # Run pre-tokenization on your chunk and store the counts for each pre-token

    # Vocabulary Initialization:
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for (i, special_token) in enumerate(special_tokens):
        vocab[i + 256] = special_token.encode("utf-8")

    # Pre-tokenization
    word_counts = get_pre_tokenized_data(input_path, special_tokens)

    # Keep track of the values that were merged
    merges: list[tuple[bytes, bytes]] = []

    # Iteratively merge until our vocab size is reached
    while len(vocab) < vocab_size:

        # Count highest pairs
        # Merge all instances of highest pairs
        pair_counter = Counter()

        # Go through every word
        for (word, count) in word_counts.items():

            # Go through each pair within the word
            # Increment counter for pair according to instances of word
            for pair in zip(word, word[1:]):
                pair_counter[pair] += count

        # Add the most common pair as our newest merge
        top_counted_pair = max(pair_counter.items(), key=lambda pair_count: (pair_count[1], pair_count[0]))[0]
        merges.append(top_counted_pair)
        
        # Merge all instances of this pair within the pre_tokenized_data
        merge = b''.join(top_counted_pair)
        new_word_counts: Counter[tuple[bytes]] = Counter()
        for (word, count) in word_counts.items():
            new_word = []
            i = 0
            while i < len(word):
                if i+1 < len(word) and (word[i], word[i+1]) == top_counted_pair:
                   new_word.append(merge)
                   i += 2
                else:
                   new_word.append(word[i])
                   i += 1
            new_word_counts[tuple(new_word)] = count
        
        word_counts = new_word_counts

        # Add our new merge to the vocab
        vocab[len(vocab)] = merge
    
    return (vocab, merges)


def get_pre_tokenized_data(input_path: str |os.PathLike, special_tokens: list[str]) -> Counter[tuple[bytes]]:
    """
    Splits a corpus into pretokens (to be further tokenized by BPE).
    """

    num_processes = cpu_count()

    with open(input_path, "rb") as f:

        boundaries = find_chunk_boundaries(
            f, num_processes, special_tokens[0].encode("utf-8")
        )

        args = []
        for i in range(len(boundaries) - 1):
            args.append((input_path, special_tokens, boundaries[i], boundaries[i + 1]))

        with Pool(num_processes) as pool:
            results = pool.map(process_chunk, args)

        aggregated_counter = Counter()

        for counter in results:
            aggregated_counter.update(counter)

        return aggregated_counter
    

def process_chunk(args: tuple[str, list[str], int, int]) -> Counter[tuple[bytes]]: 
    """
    Pretokenize a single chunk of text and return the counted words.
    """
    input_path, special_tokens, begin_index, end_index = args

    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    escaped_special_tokens = "|".join(escaped_special_tokens)
    compiled_escaped_special_tokens = re.compile(f"({escaped_special_tokens})")

    with open(input_path, "br") as f:
        f.seek(begin_index)
        chunk_text = f.read(end_index - begin_index).decode("utf-8", errors="ignore")

        counted_words: Counter[tuple[bytes]] = Counter()
        split_text = compiled_escaped_special_tokens.split(chunk_text)
        for split in split_text:
            if split not in special_tokens and split.strip():
                for match in COMPILED_PRE_TOKENIZATION_REGEX.finditer(split):
                    counted_words.update([tuple(bytes([b]) for b in match.group().encode("utf-8"))])

        return counted_words


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

if __name__ == "__main__":

    train_bpe(
        "/Users/michaelferris/recurse/ai/cs336/assignment1-basics/tests/fixtures/corpus.en",
        500,
        ["<|endoftext|>"],
    )
