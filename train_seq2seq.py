import torch
import torch.nn as nn
import os
import numpy as np

# ----------------------------
# Seq2Seq classes (same as your current model)
# ----------------------------
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, (hidden, cell)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_token, hidden, cell):
        embedded = self.embedding(input_token).unsqueeze(1)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        seq_len = trg.size(1) if trg is not None else src.size(1)
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, seq_len, vocab_size).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(src)

        input_token = torch.zeros(batch_size, dtype=torch.long).to(self.device)  # <BOS>

        for t in range(seq_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = output
            if trg is not None and np.random.rand() < teacher_forcing_ratio:
                input_token = trg[:, t]  # teacher forcing
            else:
                input_token = output.argmax(1)
        return outputs

# ----------------------------
# Load features
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_dir = "../MLDS_hw2_1_data/feature"

feature_files = sorted([
    os.path.join(feature_dir, f)
    for f in os.listdir(feature_dir)
    if f.endswith('.npy')
])

all_features = [torch.tensor(np.load(ff), dtype=torch.float32) for ff in feature_files]

# Pad sequences to same length
seq_lens = [v.shape[0] for v in all_features]
max_len = max(seq_lens)
batch = []
for v in all_features:
    if v.shape[0] < max_len:
        pad = torch.zeros(max_len - v.shape[0], v.shape[1])
        v = torch.cat([v, pad], dim=0)
    batch.append(v)
src = torch.stack(batch).to(device)

# ----------------------------
# Load captions (integer tokens)
# ----------------------------
with open("../MLDS_hw2_1_data/captions.txt") as f:
    lines = f.readlines()
# Each line should be space-separated integers
trg = torch.tensor([list(map(int, line.strip().split())) for line in lines], dtype=torch.long).to(device)

# ----------------------------
# Initialize model
# ----------------------------
input_size = all_features[0].shape[1]
vocab_size = trg.max().item() + 1  # total number of tokens
encoder = EncoderRNN(input_size, hidden_size=256)
decoder = DecoderRNN(hidden_size=256, output_size=vocab_size)
model = Seq2Seq(encoder, decoder, device).to(device)

# ----------------------------
# Training setup
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 5  # adjust as needed

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(src, trg)
    # reshape for cross-entropy: (batch*seq_len, vocab_size)
    loss = criterion(output.view(-1, vocab_size), trg.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ----------------------------
# Save trained model
# ----------------------------
torch.save(model.state_dict(), "seq2seq_trained.pt")
print("Training complete. Model saved as seq2seq_trained.pt")

