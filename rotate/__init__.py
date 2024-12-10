import cv2
import numpy as np
from PIL import Image
import pytesseract

def zoom(image):
    edges = cv2.Canny(image, 100, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def detect_dark_square_position(image, save_path):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped_image = zoom(gray)
    
    # cv2.imwrite(save_path + "_0_crop.jpg", cropped_image)
    
    margin_top = height // 10
    margin_bottom = -height // 10
    margin_left = width // 10
    margin_right = -width // 10
    
    # cv2.imshow("", cropped_image[0:margin_top, 0:margin_left])
    # cv2.waitKey(0)
    # cv2.imshow("", cropped_image[0:margin_top, margin_right:])
    # cv2.waitKey(0)
    # cv2.imshow("", cropped_image[margin_bottom:, 0:margin_left])
    # cv2.waitKey(0)
    # cv2.imshow("", cropped_image[margin_bottom:, margin_right:])
    # cv2.waitKey(0)
    
    quarters = {
        "top-left":     cropped_image[0:margin_top, 0:margin_left],
        "top-right":    cropped_image[0:margin_top, margin_right:],
        "bottom-left":  cropped_image[margin_bottom:, 0:margin_left],
        "bottom-right": cropped_image[margin_bottom:, margin_right:]
    }

    avg_intensities = {position: np.mean(quarter) for position, quarter in quarters.items()}
    
    # for p,i in avg_intensities.items():
    #     print(f"{p} = {i}")

    darkest_quarter = min(avg_intensities, key=avg_intensities.get)
    
    # print(f"Darkest Quarter = {darkest_quarter}")

    if darkest_quarter in ["top-right", "bottom-left"]:
        return True
    else:
        return False


def check_bottom_left_with_ocr(image):
    # Crop the bottom-left quarter of the image
    # Assuming rotated_image is a cv2 image (NumPy array)
    height, width, _ = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    norm = cv2.normalize(gray, np.zeros((image.shape[0], image.shape[1])), 0, 255, cv2.NORM_MINMAX)

    # Crop the bottom-left quarter of the image
    crop = norm[height // 2:height, 0:width // 2]

    # Resize the cropped region (increasing its size by a factor of 2)
    enlarge = cv2.resize(crop, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

    _, th = cv2.threshold(enlarge, 165, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("", th)
    cv2.waitKey(0)


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


def rotate_chessboard(image_path, save_path):
    # Load the image
    image = cv2.imread(image_path)

    is_dark_square = detect_dark_square_position(image, save_path)

    # Rotate the image
    if not is_dark_square:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        print("Rotate 90 degrees")
    else:
        rotated_image = image

    cv2.imwrite(save_path + "_1_corner.jpg", rotated_image)

    # Verify bottom-left corner with OCR
    if check_bottom_left_with_ocr(rotated_image):
        rotated_image_2 = rotated_image
    else:
        # Rotate 180 degrees more if needed
        rotated_image_2 = cv2.rotate(rotated_image, cv2.ROTATE_180)
        print("Rotate 180 degrees")

    cv2.imwrite(save_path+ "_2_swap.jpg", rotated_image_2)
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
            output_file_path = os.path.join("rotate/images/after", file[:-4])
            
            # Apply the auto-rotate chessboard function
            print(f"Processing {file}...")
            rotate_chessboard(file_path, output_file_path)
            print("Done")
            print()
