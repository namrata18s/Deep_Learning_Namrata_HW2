import math

# --- Helper functions from your bleu_eval.py ---

def count_ngram(candidate, references, n):
    # Example for unigram BLEU; adjust if your original function is different
    c = len(candidate.split())
    r = min(len(ref.split()) for ref in references)
    bp = brevity_penalty(c, r)
    return c, bp

def brevity_penalty(c, r):
    return 1 if c > r else math.exp(1 - float(r)/c) if c > 0 else 0  # avoid div by zero

def BLEU(candidate, references):
    # Example wrapper; replace with your actual BLEU logic
    pr, bp = count_ngram(candidate, references, 1)
    return bp  # simple example

# --- Load predictions ---
with open("test_output_text.txt", "r") as f:
    lines = f.readlines()

# Clean empty lines
lines = [line.strip() for line in lines if line.strip()]

# Example references: replace with actual reference captions list per line
references = [["A man and woman are eating at a table."]] * len(lines)  # dummy placeholder

# Compute BLEU scores safely
bleu_scores = []
for pred, refs in zip(lines, references):
    if len(pred) == 0:
        continue
    bleu_scores.append(BLEU(pred, refs))

average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
print("Average BLEU score:", average_bleu)

