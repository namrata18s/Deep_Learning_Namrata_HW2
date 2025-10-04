import json
import os
import numpy as np

# Load captions
with open("train_caption.json", "r") as f:
    captions = json.load(f)

# Convert list of dicts to a dict for easy access
captions_dict = {item['video_id']: item['captions'] for item in captions}

# Example: check if a video id exists and print first 3 captions
vid_id = "xBePrplM4OA_6_18.avi"
if vid_id in captions_dict:
    print("[INFO] Captions:", captions_dict[vid_id][:3])
else:
    print(f"[WARNING] Video id {vid_id} not found in captions.")

# Check feature files
feature_files = [f for f in os.listdir() if f.endswith(".npy")]
print("[INFO] Found feature files:", feature_files)

# Check shape of a sample feature
if feature_files:
    sample_feature = np.load(feature_files[0])
    print("[INFO] Shape of sample feature:", sample_feature.shape)
