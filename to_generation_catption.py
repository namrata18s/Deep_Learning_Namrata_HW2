import numpy as np

num_videos = 1550        # number of .npy feature files
max_caption_len = 10     # number of tokens per caption
vocab_size = 20          # token IDs range from 0 to 19

# Generate random token sequences
dummy_captions = np.random.randint(0, vocab_size, size=(num_videos, max_caption_len))

# Save to captions.txt
with open("../MLDS_hw2_1_data/captions.txt", "w") as f:
    for cap in dummy_captions:
        f.write(" ".join(map(str, cap)) + "\n")

print("Dummy captions.txt generated with", num_videos, "lines.")

