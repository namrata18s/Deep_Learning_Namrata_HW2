import json
import os
import numpy as np

# ---------------------------
# Load captions
# ---------------------------
caption_file = "train_caption.json"
if not os.path.exists(caption_file):
    raise FileNotFoundError(f"{caption_file} not found!")

with open(caption_file, "r") as f:
    captions = json.load(f)

# Convert list of dicts to a dictionary for easy access
captions_dict = {item['id']: item['caption'] for item in captions}
print(f"[INFO] Loaded {len(captions_dict)} videos from captions")

# ---------------------------
# Load feature files
# ---------------------------
feature_dir = "feature"
if not os.path.exists(feature_dir):
    raise FileNotFoundError(f"Feature folder '{feature_dir}' not found!")

feature_files = [f for f in os.listdir(feature_dir) if f.endswith(".npy")]
print(f"[INFO] Found {len(feature_files)} feature files in '{feature_dir}'")

# ---------------------------
# Display captions and check features
# ---------------------------
missing_captions = []

for f in feature_files:
    vid_id = f.replace(".npy", "")
    if vid_id in captions_dict:
        caps = captions_dict[vid_id]
        print(f"\n[INFO] Video ID: {vid_id}")
        print("[INFO] Number of captions:", len(caps))
        print("[INFO] First 3 captions:", caps[:3])
    else:
        missing_captions.append(vid_id)
        print(f"\n[WARNING] No captions found for video ID: {vid_id}")

# ---------------------------
# Check shape of a sample feature
# ---------------------------
if feature_files:
    sample_feature = np.load(os.path.join(feature_dir, feature_files[0]))
    print(f"\n[INFO] Shape of sample feature '{feature_files[0]}': {sample_feature.shape}")

# ---------------------------
# Summary of missing captions
# ---------------------------
print(f"\n[INFO] Total feature files without captions: {len(missing_captions)}")
if missing_captions:
    print("[INFO] Example missing IDs:", missing_captions[:5])

