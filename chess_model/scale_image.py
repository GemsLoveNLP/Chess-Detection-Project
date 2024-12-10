from PIL import Image, ImageOps
import os

# Input directory containing the .png images
input_directory = "E:/split_data/train/images"
# Output directory to save the .jpg images
output_directory = "C:/Desktop/Chess-Detection-Project/chess_model/chess_data/train/images/"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Target dimensions
target_size = (416, 416)

# Iterate through all files in the input directory
for filename in os.listdir(input_directory):
    if filename.lower().endswith(".png"):  # Check if the file is a .png image
        input_path = os.path.join(input_directory, filename)
        
        # Open and process the image
        with Image.open(input_path) as img:
            # Resize the image to the target size
            resized_img = img.resize(target_size, Image.BILINEAR)

            # Convert and save as .jpg
            output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.jpg")
            resized_img.convert("RGB").save(output_path, "JPEG")
        
        print(f"Processed and saved: {output_path}")


print("All images have been processed, rotated (if needed), and PNG files deleted.")

