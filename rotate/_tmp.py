import pytesseract
import cv2
import numpy as np

# Load the image
image_path = 'rotate/images/after/2_1_corner.jpg'
image = cv2.imread(image_path)


# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# height, width, _ = gray.shape
# enlarge = cv2.resize(gray, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
# height, width = enlarge.shape
# norm = cv2.normalize(enlarge, np.zeros((height, width)), 0, 255, cv2.NORM_MINMAX)
# _, th = cv2.threshold(norm, 165, 255, cv2.THRESH_BINARY)
# dilated_image = cv2.erode(th, np.ones((2, 2), np.uint8), iterations=2)

# crop = dilated_image[height // 2:, 0:width // 2]
# crop_left = dilated_image[height // 2:, 0:width // 5]
# crop_bot = dilated_image[-height // 5:, 0:width // 2]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = gray.shape
crop = gray[height // 2:, 0:width // 2]
height, width = crop.shape
enlarge = cv2.resize(crop, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
height, width = enlarge.shape
norm = cv2.normalize(enlarge, np.zeros((height, width)), 0, 255, cv2.NORM_MINMAX)
th = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 15)

# dilate = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=1)
erode = cv2.erode(th, np.ones((3, 3), np.uint8), iterations=1)



custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=12345678abcdefgh'

extracted_text = pytesseract.image_to_string(erode, config=custom_config)
print(f"Extracted Text: {extracted_text.split()}")
cv2.imshow("", erode)
cv2.waitKey(0)



# extracted_text = pytesseract.image_to_string(crop_left, config=custom_config)
# print(f"Extracted Text Left: {extracted_text.split()}")
# cv2.imshow("", crop_left)
# cv2.waitKey(0)


# extracted_text = pytesseract.image_to_string(crop_bot, config=custom_config)
# print(f"Extracted Text Bottom: {extracted_text.split()}")
# cv2.imshow("", crop_bot)
# cv2.waitKey(0)

# flipped = cv2.flip(th, 0)

# cv2.imshow("", flipped)
# cv2.waitKey(0)