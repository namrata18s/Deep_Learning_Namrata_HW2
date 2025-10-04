# Video Caption Generation with Sequence-to-Sequence and Attention

## Overview

This project implements a **video captioning system** using a **sequence-to-sequence (seq2seq) architecture** with LSTM-based encoder-decoder networks and an attention mechanism. The system generates descriptive captions for short video clips, following modern deep learning practices.

---

## Dataset

- **Dataset:** MSVD (preprocessed as provided in the assignment).  
- **Training Data:**  
  - Video features: `.npy` files located in `training_data/feat/`.  
  - Captions: `train_caption.json`.  
- **Preprocessing:**  
  - Features aligned with captions using video IDs.  
  - Vocabulary includes `<PAD>`, `<BOS>`, `<EOS>`, and `<UNK>`.  
  - Variable sequence lengths are handled with padding.

---

## Model Architecture

- **Encoder:** LSTM network processes video features into hidden representations.  
- **Decoder:** LSTM generates captions word-by-word, conditioned on encoder outputs.  
- **Attention Mechanism:** Helps the decoder focus on relevant video frames.  
- **Scheduled Sampling:** Sometimes uses predicted words as next inputs during training to reduce exposure bias.  

**Hyperparameters:**
- Hidden dimension: 256  
- Embedding dimension: 256  
- Training epochs: 20  

---

## Training

- **Loss Function:** Cross-entropy (ignoring padding tokens)  
- **Optimizer:** Adam (learning rate: 0.001)  
- **Batching:** `torch.utils.data.DataLoader`  
- **Inference:** Beam search with width 3  

**Challenges & Solutions:**
- Constructed `VideoCaptionDataset` for robust data loading and alignment.  
- Handled flexible file paths for reproducibility.  
- Optimized memory and batching for efficient training.

---

## Results

- The model gradually improves captions during training.  
- BLEU-1 scores (validation) approach or exceed the baseline of 0.6.  

**Example Predictions:**

| Video ID | Ground Truth Caption | Predicted Caption |
|----------|-------------------|-----------------|
| xyz123   | A man is playing guitar | A person plays guitar |
| abc456   | A dog is running in the park | Dog runs in park |

---

## Usage

### Training
```bash
python3 modelseq2seq.py --mode train --data_dir MLDS_hw2_1_data/training_data --epochs 20
