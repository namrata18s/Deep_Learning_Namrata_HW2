# decode_test_output_quick.py
input_file = "test_output_fixed.txt"
output_file = "test_output_decoded.txt"

with open(input_file) as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        # Split at the comma to skip video ID
        parts = line.strip().split(",", 1)
        if len(parts) < 2:
            continue
        caption_ids = parts[1]
        # Just write the IDs as they are (or optionally map to dummy tokens)
        f_out.write(caption_ids + "\n")

print(f"Decoded captions written to {output_file}!")

