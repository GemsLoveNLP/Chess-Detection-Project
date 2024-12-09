import cv2
import numpy as np
from PIL import Image
import pytesseract

def detect_dark_square_position(image):
    """
    Detects if the darkest square is in the top-right or bottom-left corners 
    by comparing the average intensity of regions 20% from the borders.

    Args:
        image: Input image as a NumPy array.

    Returns:
        bool: True if the darkest quarter is in the top-right or bottom-left, False otherwise.
    """
    # Resize image for easier processing
    resized = cv2.resize(image, (400, 400))
    height, width = resized.shape[:2]

    # Convert the blurred image to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]

    # Calculate the 20% border margin
    margin_top = int(height * 0.2)
    margin_bottom = int(height * 0.8)
    margin_left = int(width * 0.2)
    margin_right = int(width * 0.8)

    # Define regions (quarters) that are 20% from the image border
    quarters = {
        "top-left": cropped_image[0:margin_top, 0:margin_left],
        "top-right": cropped_image[0:margin_top, margin_right:width],
        "bottom-left": cropped_image[margin_bottom:height, 0:margin_left],
        "bottom-right": cropped_image[margin_bottom:height, margin_right:width]
    }

    # Analyze each quarter for the average intensity (lower intensity means darker square)
    avg_intensities = {}
    for position, quarter in quarters.items():
        avg_intensities[position] = np.mean(quarter)

    # Print each corner's average intensity for debugging
    # for position, intensity in avg_intensities.items():
    #     print(f"Position: {position}, Intensity: {intensity}")

    # Find the quarter with the lowest average intensity (darkest quarter)
    darkest_quarter = min(avg_intensities, key=avg_intensities.get)

    # print(f"Darkest quarter: {darkest_quarter}")

    # Return True if the darkest quarter is in top-right or bottom-left, False otherwise
    if darkest_quarter in ["top-right", "bottom-left"]:
        return True
    else:
        return False


def check_bottom_left_with_ocr(image):
    """
    Verifies if the bottom-left corner contains labels indicating correct orientation 
    ('a', 'b', 'c', '1', '2', '3') for proper rotation. If incorrect ('f', 'g', 'h', '6', '7', '8'), 
    additional rotation is applied.

    Args:
        rotated_image: The rotated Image object.

    Returns:
        bool: True if the orientation is correct (no further rotation needed), False if 180° rotation is required.
    """
    # Crop the bottom-left quarter of the image
    # Assuming rotated_image is a cv2 image (NumPy array)
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    norm = cv2.normalize(gray, np.zeros((image.shape[0], image.shape[1])), 0, 255, cv2.NORM_MINMAX)

    # Crop the bottom-left quarter of the image
    crop = norm[height // 2:height, 0:width // 2]

    # Resize the cropped region (increasing its size by a factor of 2)
    enlarge = cv2.resize(crop, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

    # kernel = np.array([[-1, -1, -1],
    #                           [-1,  9, -1],
    #                           [-1, -1, -1]])
    # sharp = cv2.filter2D(enlarge, -1, kernel)

    # _, th = cv2.threshold(enlarge, 170, 255, cv2.THRESH_TOZERO)
    _, th = cv2.threshold(enlarge, 165, 255, cv2.THRESH_BINARY)



    cv2.imshow("", th)
    cv2.waitKey(2000)
    
    # Perform OCR on the cropped region
    detected_text = pytesseract.image_to_string(
        th,
        config='--psm 11 -c tessedit_char_whitelist=12345678abcdefgh'
        # config='--psm 3 -c tessedit_char_whitelist=12345678abcdefgh'
        )
    print("OCR detected in bottom-left quarter:", detected_text.split())

    # Check for correct orientation indicators
    correct_numbers = ['1', '2', '3', '4']
    incorrect_numbers = ['5', '6', '7', '8']
    correct_letters = ['b', 'c', 'd']
    incorrect_letters = ['e', 'f', 'g', 'h']

    # Check orientation based on numbers
    if any(number in detected_text.lower() for number in correct_numbers):
        return True  # Correct orientation based on numbers
    elif any(number in detected_text.lower() for number in incorrect_numbers):
        return False  # Needs 180° rotation based on numbers

    # Check orientation based on letters
    if any(letter in detected_text.lower() for letter in correct_letters):
        return True  # Correct orientation based on letters
    elif any(letter in detected_text.lower() for letter in incorrect_letters):
        return False  # Needs 180° rotation based on letters

    # If no indicators are detected
    print("No clear orientation detected. Assuming image is correct.")
    return True


def auto_rotate_chessboard(image_path, save_path):
    """
    Automatically rotates the chessboard image so that 'a1' is in the bottom-left corner using both
    dark square detection and OCR verification.

    Args:
        image_path (str): Path to the input image.
        save_path (str): Path to save the corrected image.
    """
    # Load the image
    image = cv2.imread(image_path)

    is_dark_square = detect_dark_square_position(image)

    # Determine initial rotation based on the dark square's position
    # rotation_angle = 0 if is_dark_square else 90

    # Rotate the image
    if not is_dark_square:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        print("Rotate 90 degrees")
    else:
        rotated_image = image  # No rotation

    # Verify bottom-left corner with OCR
    if check_bottom_left_with_ocr(rotated_image):
        rotated_image_2 = rotated_image
    else:
        # Rotate 180 degrees more if needed
        rotated_image_2 = cv2.rotate(rotated_image, cv2.ROTATE_180)
        print("Rotate 180 degrees")

    cv2.imwrite(save_path, rotated_image_2)
    print(f"Final image saved at: {save_path}")



# Example usage
if __name__ == "__main__":
    import os
    # dir = "chess_model/chess_data/train/images"
    dir = "rotate/images/before"
    files = os.listdir(dir)

    for file in files:
        file_path = os.path.join(dir, file)
        
        # Check if the file is an image (e.g., jpg, png, jpeg)
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            output_file_path = os.path.join("rotate/images/after", f"rotated_{file}")
            
            # Apply the auto-rotate chessboard function
            print(f"Processing {file}...")
            auto_rotate_chessboard(file_path, output_file_path)
            print("Done")
            print()
