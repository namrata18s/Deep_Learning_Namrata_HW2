import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import random
from collections import Counter

# Model Definitions

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        outputs, (h, c) = self.lstm(x)
        return outputs, h, c

class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.attn = nn.Linear(enc_dim + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1, bias=False)
    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attn_weights = torch.softmax(self.v(energy).squeeze(2), dim=1)
        return attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_dim, dec_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + enc_dim, dec_dim, batch_first=True)
        self.fc = nn.Linear(dec_dim, vocab_size)
        self.attention = Attention(enc_dim, dec_dim)
    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.embedding(input)
        attn_weights = self.attention(hidden[-1].unsqueeze(1), encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        pred = self.fc(output.squeeze(1))
        return pred, hidden, cell, attn_weights

# Vocabulary Building

def build_vocab(sentences, min_count=3):
    counter = Counter(word for sent in sentences for word in sent.split())
    specials = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
    vocab = specials + [w for w, c in counter.items() if c >= min_count]
    word2idx = {tok: i for i, tok in enumerate(vocab)}
    idx2word = {i: tok for tok, i in word2idx.items()}
    return word2idx, idx2word

# Dataset Stub (Replace with your data loading logic)

class VideoCaptionDataset(Dataset):
    def __init__(self, data_dir, vocab, max_caption_len=30):
        self.feat_dir = os.path.join(data_dir, 'feat')
        import os
        import json

        caption_json_path = "../MLDS_hw2_1_data/train_caption.json"

        with open(json_path, 'r') as f:
            captions_data = json.load(f)


        self.vocab = vocab
        self.max_caption_len = max_caption_len
        
        self.samples = []
        for item in captions_data:
            video_id = item['id']
            caption = item['caption']
            feat_path = os.path.join(self.feat_dir, f"{video_id}.npy")
            if os.path.exists(feat_path):
                self.samples.append((feat_path, caption))
        
    def __len__(self):
        return len(self.samples)

    def tokenize_caption(self, caption):
        tokens = caption.lower().strip().split()
        tokens = ['<BOS>'] + tokens + ['<EOS>']
        token_ids = [self.vocab.get(tok, self.vocab['<UNK>']) for tok in tokens]
        
        if len(token_ids) < self.max_caption_len:
            token_ids += [self.vocab['<PAD>']] * (self.max_caption_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_caption_len]
        return torch.tensor(token_ids, dtype=torch.long)

    def __getitem__(self, idx):
        feat_path, caption = self.samples[idx]
        video_feat = torch.tensor(np.load(feat_path), dtype=torch.float32)
        caption_tokens = self.tokenize_caption(caption)
        return video_feat, caption_tokens


# Training Function with Schedule Sampling

def train_epoch(encoder, decoder, dataloader, criterion, optimizer, vocab, device, schedule_sampling_prob=0.1):
    encoder.train()
    decoder.train()
    total_loss = 0
    for video_feats, captions in dataloader:
        video_feats, captions = video_feats.to(device), captions.to(device)
        optimizer.zero_grad()

        encoder_outputs, h, c = encoder(video_feats)
        batch_size = video_feats.size(0)
        input_word = torch.full((batch_size, 1), vocab['<BOS>'], dtype=torch.long).to(device)

        loss = 0
        for t in range(1, captions.size(1)):
            output, h, c, _ = decoder(input_word, h, c, encoder_outputs)
            loss += criterion(output, captions[:, t])
            use_sampling = random.random() < schedule_sampling_prob
            top1 = output.argmax(1).unsqueeze(1)
            input_word = top1 if use_sampling else captions[:, t].unsqueeze(1)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Beam Search for Inference

def beam_search(encoder, decoder, video_feats, vocab, idx2word, device, beam_width=3, max_len=20):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        encoder_outputs, h, c = encoder(video_feats.to(device))
        sequences = [[[], 0.0, h, c]]

        for _ in range(max_len):
            all_candidates = []
            for seq, score, h, c in sequences:
                input_word = torch.tensor([vocab[seq[-1]]] if seq else vocab['<BOS>']).unsqueeze(1).to(device)
                output, h, c, _ = decoder(input_word, h, c, encoder_outputs)
                probs = torch.softmax(output, dim=1).squeeze(0)
                topk = torch.topk(probs, beam_width)
                for i in range(beam_width):
                    candidate = seq + [topk.indices[i].item()]
                    candidate_score = score - torch.log(topk.values[i])
                    all_candidates.append([candidate, candidate_score, h, c])
            sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

        # Convert token ids to words and stop at EOS
        best_seq = sequences[0][0]
        words = []
        for idx in best_seq:
            if idx == vocab['<EOS>']:
                break
            if idx not in [vocab['<PAD>'], vocab['<BOS>']]:
                words.append(idx2word.get(idx, '<UNK>'))
        return ' '.join(words)

# Main CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output', type=str)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Replace with your actual data loading and vocabulary building
    word2idx, idx2word = build_vocab([])  # Empty for example; load real captions here

    encoder = Encoder(input_dim=1024, hidden_dim=256).to(device)
    decoder = Decoder(vocab_size=len(word2idx), embed_dim=256, enc_dim=256, dec_dim=256).to(device)

    if args.mode == 'train':
        import json

        with open('MLDS_hw2_1_data/train_caption.json', 'r') as f:
            captions_data = json.load(f)
        all_captions = [item['caption'] for item in captions_data]

        vocab, idx2word = build_vocab(all_captions, min_count=3)

        dataset = VideoCaptionDataset(args.data_dir, vocab)

        dataset = VideoCaptionDataset('MLDS_hw2_1_data/training_data', vocab)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
        for epoch in range(args.epochs):
            loss = train_epoch(encoder, decoder, dataloader, criterion, optimizer, word2idx, device)
            print(f'Epoch {epoch+1}/{args.epochs} loss: {loss:.4f}')
        torch.save(encoder.state_dict(), 'encoder.pth')
        torch.save(decoder.state_dict(), 'decoder.pth')

    elif args.mode == 'test':
        encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
        decoder.load_state_dict(torch.load('decoder.pth', map_location=device))
        dataset = VideoCaptionDataset('MLDS_hw2_1_data/training_data', vocab)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        with open(args.output, 'w') as f_out:
            for video_feats, _ in dataloader:
                caption = beam_search(encoder, decoder, video_feats, word2idx, idx2word, device)
                f_out.write(caption + '\n')
        print(f'Captions written to {args.output}')

if __name__ == '__main__':
    main()
