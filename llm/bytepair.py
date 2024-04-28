import argparse

# Inspiration from this was drawn from the "Let's build the GPT Tokenizer" video by Andrej Karpathy.

MIT_LICENSE = """
 Permission is hereby granted, free of charge, to any person
 obtaining a copy of this software and associated documentation
 files (the "Software"), to deal in the Software without
 restriction, including without limitation the rights to use,
 copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following
 conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 OTHER DEALINGS IN THE SOFTWARE.
    """


def most_common_pair(encoding):
    """Retrieves the most common pair.

    Note:
        There is actually an interesting edge case here which I did not find mentioned online. Concretely, in the
        sequence "aaa" we will count "aa" twice. However, in the merging phase, this can only be merged once. I wonder
        whether this is just left as is or if people check for this edge case in practice.

    """
    pairs = {}
    for idx in range(len(encoding) - 1):
        pair = (encoding[idx], encoding[idx + 1])
        pairs[pair] = pairs.get(pair, 0) + 1
    return max(pairs, key=pairs.get)


def merge(encoding, pair_to_merge, new_token):
    """Merges the new pair into the tokens."""
    new_encoding = []
    idx = 0
    while idx < len(encoding):
        """We cannot just iterate until len(encoding) -1 since we need to also add the final token"""
        if idx < len(encoding) - 1 and encoding[idx] == pair_to_merge[0] and encoding[idx + 1] == pair_to_merge[1]:
            new_encoding.append(new_token)
            idx += 2
        else:
            new_encoding.append(encoding[idx])
            idx += 1
    return new_encoding


def byte_pair_decoding(encoding, vocab):
    """Implements the decoding part of the tokenization process.

    Note:
        This essentially reverses the encoding step. However, it can be done faster by reversing the vocab list and
        iterating through the encoding list once.
    """
    data = encoding
    for new_token, pair in reversed(vocab):
        new_data = []
        for token in data:
            if token == new_token:
                new_data.extend(list(pair))
            else:
                new_data.append(token)
        data = new_data
    data = ''.join([chr(c) for c in data])
    return data


def byte_pair_decoding_fast(encoding, vocab):
    """Implements the fast version of the decoder."""
    data = encoding
    vocab_dict = {idx: [idx] for idx in range(256)}
    for new_token, pair in vocab:
        vocab_dict[new_token] = [t for p in [pair[0], pair[1]] for t in vocab_dict[p]]
    new_data = [char for token in data for char in vocab_dict[token]]
    data = ''.join([chr(c) for c in new_data])
    return data


def byte_pair_encoding(data, vocab_size):
    """Implements the byte pair encoding strategy used for tokenization in LLMs."""
    encoding = [ord(c) for c in data]
    vocab = []
    num_merges = vocab_size - 256  # The 256 is the vocabulary size of UTF-8.
    for i in range(num_merges):
        pair_to_merge = most_common_pair(encoding)
        new_token = 256 + i
        vocab.append((new_token, pair_to_merge))
        encoding = merge(encoding, pair_to_merge, new_token)
    print(f'Compression ratio: {len(data) / len(encoding)}')
    return encoding, vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=300)
    args = parser.parse_args()

    encoding, vocab = byte_pair_encoding(MIT_LICENSE, args.vocab_size)
    decoding = byte_pair_decoding(encoding, vocab)
    decoding_faster = byte_pair_decoding_fast(encoding, vocab)
    print(f'Successful decoding: {MIT_LICENSE == decoding}')
    print(f'Successful faster decoding: {MIT_LICENSE == decoding_faster}')
