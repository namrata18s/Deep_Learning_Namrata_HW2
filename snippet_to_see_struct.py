import pickle

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

print(type(vocab))
print(dir(vocab))  # list all attributes
print(vocab.__dict__ if hasattr(vocab, "__dict__") else vocab)

