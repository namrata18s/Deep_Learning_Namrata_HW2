import torch
import torch.nn as nn
import os
import numpy as np
import sys

# ----------------------------
# Seq2Seq classes
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

    def forward(self, src, trg=None, teacher_forcing_ratio=0.0):
        batch_size = src.size(0)
        seq_len = src.size(1)
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, seq_len, vocab_size).to(self.device)

        encoder_outputs, (hidden, cell) = self.encoder(src)
        input_token = torch.zeros(batch_size, dtype=torch.long).to(self.device)  # <BOS>

        for t in range(seq_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = output
            input_token = output.argmax(1)  # greedy decoding
        return outputs

# ----------------------------
# Load features
# ----------------------------
if __name__ == "__main__":
    feature_dir = sys.argv[1]      # e.g., ../MLDS_hw2_1_data/feature
    output_file = sys.argv[2]       # e.g., test_output.txt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_files = sorted([
        os.path.join(feature_dir, f)
        for f in os.listdir(feature_dir)
        if f.endswith('.npy')
    ])

    if len(feature_files) == 0:
        print(f"No .npy feature files found in {feature_dir}. Exiting.")
        sys.exit(1)

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
    # Initialize and load trained model
    # ----------------------------
    input_size = all_features[0].shape[1]
    vocab_size = 20  # same as your trained model
    encoder = EncoderRNN(input_size, hidden_size=256)
    decoder = DecoderRNN(hidden_size=256, output_size=vocab_size)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Load trained weights
    model.load_state_dict(torch.load("seq2seq_trained.pt"))
    model.eval()

    # ----------------------------
    # Generate captions
    # ----------------------------
    with torch.no_grad():
        output = model(src)
        captions = output.argmax(-1).tolist()

    # Write captions to file
    with open(output_file, "w") as f:
        for cap in captions:
            f.write(" ".join(map(str, cap)) + "\n")

    print(f"Captions written to {output_file}!")
