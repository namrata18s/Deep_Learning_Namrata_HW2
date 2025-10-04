# decode_captions_simple.py
import sys

if len(sys.argv) < 2:
    print("Usage: python decode_captions_simple.py <caption_file>")
    sys.exit(1)

caption_file = sys.argv[1]

with open(caption_file, "r") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        # If you have video IDs like in test_output_fixed.txt
        if ',' in line:
            vid, caption = line.split(',', 1)
        else:
            vid = f"video{i}"
            caption = line
        print(f"{vid}: {caption}")

