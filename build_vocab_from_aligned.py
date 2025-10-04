import pickle

# Load your aligned dataset
with open("aligned_dataset.pkl", "rb") as f:
    data = pickle.load(f)

# Collect all captions
all_captions = []
for item in data:
    all_captions.extend(item['captions'])

# Build vocab dictionary
vocab = {}
idx = 0
for sentence in all_captions:
    for word in sentence.strip().split():
        if word not in vocab:
            vocab[word] = idx
            idx += 1

# Save vocab as pickle
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print(f"Saved vocab with {len(vocab)} words.")

