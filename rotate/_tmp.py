import pytesseract
import cv2
import numpy as np

# Load the image
image_path = 'rotate/images/after/6_1_corner.jpg'
image = cv2.imread(image_path)

height, width, _ = image.shape
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
enlarge = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
height, width = enlarge.shape

norm = cv2.normalize(enlarge, np.zeros((height, width)), 0, 255, cv2.NORM_MINMAX)
# blurred = cv2.GaussianBlur(norm, (5, 5), 0)

_, th = cv2.threshold(norm, 165, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((2, 2), np.uint8)
dilated_image = cv2.dilate(th, kernel, iterations=2)  # You can adjust the number of iterations

crop = dilated_image[height // 2:, 0:width // 2]
crop_left = dilated_image[height // 2:, 0:width // 5]
crop_bot = dilated_image[-height // 5:, 0:width // 2]

custom_config = r'--oem 3 --psm 6'

extracted_text = pytesseract.image_to_string(crop, config=custom_config)
print(f"Extracted Text: {extracted_text.split()}")
cv2.imshow("", crop)
cv2.waitKey(0)



extracted_text = pytesseract.image_to_string(crop_left, config=custom_config)
print(f"Extracted Text Left: {extracted_text.split()}")
cv2.imshow("", crop_left)
cv2.waitKey(0)


extracted_text = pytesseract.image_to_string(crop_bot, config=custom_config)
print(f"Extracted Text Bottom: {extracted_text.split()}")
cv2.imshow("", crop_bot)
cv2.waitKey(0)

# flipped = cv2.flip(th, 0)

# cv2.imshow("", flipped)
# cv2.waitKey(0)