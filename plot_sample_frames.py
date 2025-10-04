import cv2
import random
import matplotlib.pyplot as plt
from pathlib import Path

# Number of sample videos to display
num_samples = 5

# Pick random indices
sample_indices = random.sample(range(len(preds)), num_samples)

plt.figure(figsize=(18, 6))

for i, idx in enumerate(sample_indices):
    video_file = Path(dataset[idx]['feature_file']).with_suffix('.avi')  # original video path
    cap = cv2.VideoCapture(str(video_file))
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    plt.subplot(1, num_samples, i+1)
    plt.imshow(frame)
    plt.axis('off')
    
    # Display predicted and reference captions
    pred_caption = preds[idx][:50] + '...'  # truncated predicted caption
    ref_caption = dataset[idx]['captions'][0][:50] + '...'  # first reference caption
    plt.title(f"Pred: {pred_caption}\nRef: {ref_caption}", fontsize=9)

    cap.release()

plt.tight_layout()
plt.savefig("sample_frames_with_pred_ref_captions.png", dpi=200)
plt.show()

