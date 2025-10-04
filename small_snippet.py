import pickle

with open("aligned_dataset.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))  # see if it's dict, list, etc.
if isinstance(data, dict):
    print("Keys in dict:", list(data.keys()))
elif hasattr(data, "__dict__"):
    print("Attributes:", dir(data))
else:
    print(data)

