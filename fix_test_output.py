# fix_test_output.py
with open("test_output.txt") as f_in, open("test_output_fixed.txt", "w") as f_out:
    for i, line in enumerate(f_in):
        line = line.strip()
        f_out.write(f"video{i+1},{line}\n")
print("Fixed test_output.txt → test_output_fixed.txt")

