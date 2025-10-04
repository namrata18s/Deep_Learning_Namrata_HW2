import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
import os

# ----------------------------
# Encoder & Decoder
# ----------------------------
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=0)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, (hidden, cell)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=0)
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
            input_token = output.argmax(1)  # greedy
        return outputs


# ----------------------------
# Feature extraction from video
# ----------------------------
def extract_video_features(video_path, cnn_model, device):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(video_path)
    features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = cnn_model(frame)
        features.append(feat.squeeze(0).cpu())
    cap.release()

    if len(features) == 0:
        features.append(torch.zeros(2048))
    return torch.stack(features)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1]      # e.g., ../MLDS_hw2_1_data
    output_file = sys.argv[2]   # e.g., test_output.txt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pretrained CNN (ResNet-50) for feature extraction
    cnn_model = models.resnet50(pretrained=True)
    cnn_model = nn.Sequential(*list(cnn_model.children())[:-1])  # remove classifier
    cnn_model.eval().to(device)

    # Collect all .avi videos from subfolders
    video_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".avi"):
                video_files.append(os.path.join(root, f))

    print("Found video files:", video_files)

    if len(video_files) == 0:
        print("No video files found. Exiting.")
        sys.exit(1)

    # Extract features for each video
    all_features = []
    for vf in video_files:
        feats = extract_video_features(vf, cnn_model, device)
        all_features.append(feats)

    # Pad sequences to the same length
    seq_lens = [v.shape[0] for v in all_features]
    max_len = max(seq_lens)
    batch = []
    for v in all_features:
        if v.shape[0] < max_len:
            pad = torch.zeros(max_len - v.shape[0], 2048)
            v = torch.cat([v, pad], dim=0)
        batch.append(v)
    src = torch.stack(batch).to(device)

    # Initialize seq2seq model
    encoder = EncoderRNN(input_size=2048, hidden_size=256)
    decoder = DecoderRNN(hidden_size=256, output_size=20)  # vocab_size
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Generate captions
    with torch.no_grad():
        output = model(src)
        captions = output.argmax(-1).tolist()

    # Write captions to file
    with open(output_file, "w") as f:
        for cap in captions:
            f.write(" ".join(map(str, cap)) + "\n")

    print(f"Captions written to {output_file}!")

