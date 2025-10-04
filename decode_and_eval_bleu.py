# decode_and_eval_bleu.py
import sys
import json
import math
import operator
from functools import reduce

# ----------------------------
# BLEU helper functions
# ----------------------------
def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        ref_counts = []
        ref_lengths = []
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                ngram_d[ngram] = ngram_d.get(ngram, 0) + 1
            ref_counts.append(ngram_d)
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(limits):
            ngram = ' '.join(words[i:i+n]).lower()
            cand_dict[ngram] = cand_dict.get(ngram, 0) + 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    pr = float(clipped_count)/count if count>0 else 0
    bp = brevity_penalty(c, r)
    return pr, bp

def clip_count(cand_d, ref_ds):
    count = 0
    for m, m_w in cand_d.items():
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        count += min(m_w, m_max)
    return count

def best_length_match(ref_l, cand_l):
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best

def brevity_penalty(c, r):
    return 1 if c>r else math.exp(1-(float(r)/c))

def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

def BLEU(candidate, references):
    precisions = []
    pr, bp = count_ngram(candidate, references, 1)
    precisions.append(pr)
    score = geometric_mean(precisions) * bp
    return score

# ----------------------------
# Main
# ----------------------------
if len(sys.argv) < 3:
    print("Usage: python decode_and_eval_bleu.py <caption_file> <reference_json>")
    sys.exit(1)

caption_file = sys.argv[1]
ref_json = sys.argv[2]

# Load reference captions
with open(ref_json, 'r') as f:
    references = json.load(f)

# Read predicted captions
predicted = []
with open(caption_file, 'r') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if ',' in line:
            vid, caption = line.split(',', 1)
        else:
            vid = f"video{i+1}"
            caption = line
        print(f"{vid}: {caption}")  # print decoded captions
        predicted.append(caption)

# Evaluate BLEU (simple sequential match)
bleu_scores = []
for i, item in enumerate(references):
    ref_caps = [c.rstrip('.') for c in item['caption']]
    bleu_scores.append(BLEU([predicted[i]], ref_caps))

avg_bleu = sum(bleu_scores)/len(bleu_scores)
print(f"\nAverage BLEU score: {avg_bleu:.4f}")

