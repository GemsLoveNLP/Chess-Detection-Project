import os
import re

# Directory containing the label files
directory = "E:/split_data/train/labels"

# Mapping of old numbers to new numbers (key: old number, value: new number)
label_mapping = {
    "3": "1",  # Example: Replace '7' with '2'
    "7": "2",  # Replace '9' with '4'
    "4": "3", # Replace '11' with '3'
    "6": "4",  # Example: Replace '7' with '2'
    "5": "5",  # Replace '9' with '4'
    "2": "6", # Replace '11' with '3'
    "9": "7",  # Example: Replace '7' with '2'
    "11": "8",  # Replace '9' with '4'
    "10": "9", # Replace '11' with '3'
    "1": "10",  # Example: Replace '7' with '2'
    "0": "11",  # Replace '9' with '4'
    "8": "12", # Replace '11' with '3'
}

# Iterate through all files in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    # Read the file content
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Modify the first number in each line based on the mapping
    updated_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:  # Ensure the line is not empty
            # Replace the first part (class label) if it matches a key in the mapping
            if parts[0] in label_mapping:
                parts[0] = label_mapping[parts[0]]
            updated_lines.append(" ".join(parts))

    # Write the updated content back to the file
    with open(file_path, "w") as file:
        file.write("\n".join(updated_lines))

    print(f"Updated file: {filename}")

print("All matching files have been processed.")
