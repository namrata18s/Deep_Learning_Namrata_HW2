import pickle
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

# -----------------------------
# Load aligned dataset (ground-truth captions)
# -----------------------------
with open("aligned_dataset.pkl", "rb") as f:
    data = pickle.load(f)

# -----------------------------
# Load decoded predictions
# -----------------------------
with open("test_output_text.txt", "r") as f:
    preds = [line.strip() for line in f]

# -----------------------------
# Calculate BLEU per video
# -----------------------------
bleu_scores = []
video_ids = []

for i, item in enumerate(data):
    video_ids.append(item['video_id'])
    references = [cap.split() for cap in item['captions']]  # list of reference captions (tokenized)
    
    if i < len(preds):
        candidate = preds[i].split()  # tokenized predicted caption
        score = sentence_bleu(references, candidate)
    else:
        score = 0  # in case prediction missing
    bleu_scores.append(score)

# -----------------------------
# Plot BLEU scores per video
# -----------------------------
plt.figure(figsize=(12,6))
plt.bar(video_ids, bleu_scores)
plt.xticks(rotation=90)
plt.ylabel("BLEU score")
plt.title("BLEU score per video")
plt.tight_layout()
plt.savefig("bleu_per_video.png")
plt.show()

