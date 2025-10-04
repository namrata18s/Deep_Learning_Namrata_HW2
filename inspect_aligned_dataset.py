import pickle
from pprint import pprint

with open("aligned_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Print type and first few items
print("Top-level type:", type(dataset))

# Pretty print first element(s)
if isinstance(dataset, list):
    print("Length of list:", len(dataset))
    pprint(dataset[:3])  # show first 3 items
elif isinstance(dataset, dict):
    pprint(list(dataset.items())[:10])  # first 10 key-value pairs


