import pickle

# Load aligned dataset
with open("aligned_dataset.pkl", "rb") as f:
    data = pickle.load(f)

# Extract vocab: word-to-index mapping
if 'vocab' in data:
    vocab = data['vocab']  # this should be a dict like {word: index}
elif 'word_to_idx' in data:
    vocab = data['word_to_idx']
else:
    raise ValueError("Cannot find vocab in aligned_dataset.pkl")

# Build index-to-word mapping
idx2word = {idx: word for word, idx in vocab.items()}

# Decode numeric sequences in test_output.txt
with open("test_output.txt") as f_in, open("test_output_decoded.txt", "w") as f_out:
    for i, line in enumerate(f_in):
        line = line.strip().split(",")[-1]  # skip video ID
        token_ids = [int(x) for x in line.split()]
        words = [idx2word.get(tid, "<UNK>") for tid in token_ids]
        f_out.write(f"video{i+1},{ ' '.join(words) }\n")

print("Decoded captions saved to test_output_decoded.txt")

