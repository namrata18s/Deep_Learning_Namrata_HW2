import pickle

# Load vocab mapping
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Create reverse mapping
idx2word = {idx: word for word, idx in vocab.items()}

# Optional: stop at <END> token
end_token_idx = vocab.get("<END>", None)

def remove_repeated_sequences(words):
    """Remove repeated consecutive sequences of words."""
    cleaned = []
    i = 0
    while i < len(words):
        seq_len = 1
        while i + 2*seq_len <= len(words):
            seq1 = words[i:i+seq_len]
            seq2 = words[i+seq_len:i+2*seq_len]
            if seq1 == seq2:
                i += seq_len  # skip repeated sequence
                seq_len = 1   # reset sequence length
            else:
                seq_len += 1
        cleaned.append(words[i])
        i += 1
    return cleaned

decoded_lines = []
with open("test_output_decoded.txt", "r") as f:
    for line in f:
        indices = list(map(int, line.strip().split()))
        words = []
        prev_word = None
        for i in indices:
            if end_token_idx is not None and i == end_token_idx:
                break
            if i in idx2word:
                word = idx2word[i]
                if word != prev_word:  # remove consecutive duplicates
                    words.append(word)
                prev_word = word
        # Remove repeated sequences
        words = remove_repeated_sequences(words)
        decoded_lines.append(" ".join(words))

# Save cleaned captions
with open("test_output_text.txt", "w") as f:
    for line in decoded_lines:
        f.write(line + "\n")

print("Aggressive cleanup complete!")
