import pickle

# Load vocab
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

idx2word = {v: k for k, v in vocab.items()}  # assuming vocab maps word -> index

# Load model outputs (predicted token IDs)
with open("test_output.txt", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    line = line.strip().split(",")[1]  # skip video ID
    token_ids = [int(x) for x in line.split()]  # convert to integers
    words = [idx2word.get(t, "<UNK>") for t in token_ids]
    sentence = " ".join(words)
    print(f"Video {i+1}: {sentence}")

