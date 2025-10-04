import json
import os

# ---------------------------
# Load captions
# ---------------------------
caption_file = "train_caption.json"
feature_dir = "feature"

with open(caption_file, "r") as f:
    captions = json.load(f)

# Convert to dict for easy access
captions_dict = {item['id']: item['caption'] for item in captions}

# ---------------------------
# Load feature files
# ---------------------------
feature_files = [f for f in os.listdir(feature_dir) if f.endswith(".npy")]

# ---------------------------
# Create aligned dataset
# ---------------------------
aligned_dataset = []

missing_captions = []

for f in feature_files:
    vid_id = f.replace(".npy", "")
    if vid_id in captions_dict:
        aligned_dataset.append({
            "feature_file": os.path.join(feature_dir, f),
            "video_id": vid_id,
            "captions": captions_dict[vid_id]
        })
    else:
        missing_captions.append(vid_id)

print(f"[INFO] Total aligned video-feature pairs: {len(aligned_dataset)}")
print(f"[INFO] Total feature files without captions: {len(missing_captions)}")
if missing_captions:
    print("[INFO] Example missing IDs:", missing_captions[:5])

# ---------------------------
# Optional: save aligned dataset for later use
# ---------------------------
import pickle

with open("aligned_dataset.pkl", "wb") as f:
    pickle.dump(aligned_dataset, f)

print("[INFO] Aligned dataset saved to 'aligned_dataset.pkl'")

