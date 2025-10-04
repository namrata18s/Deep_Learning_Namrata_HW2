import json
from collections import Counter

# Load train captions
with open("../MLDS_hw2_1_data/train_caption.json") as f:
    captions_data = json.load(f)

# Collect all tokens
counter = Counter()
for entry in captions_data:             # loop over list
    for sentence in entry["caption"]:   # each has "caption" list
        tokens = sentence.lower().strip().split()
        counter.update(tokens)

# Special tokens
vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}

# Add words from dataset
for word, _ in counter.most_common():
    if word not in vocab:
        vocab[word] = len(vocab)

# Save vocab
with open("../MLDS_hw2_1_data/vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

print(f"Saved vocab with {len(vocab)} words.")

