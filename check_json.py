import json

# Load the JSON file
with open("train_caption.json", "r") as f:
    captions = json.load(f)

# Print type and sample data
print("Type of captions:", type(captions))
if isinstance(captions, dict):
    print("Top-level keys (first 5):", list(captions.keys())[:5])
elif isinstance(captions, list):
    print("First item in list:", captions[0])

