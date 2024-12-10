import cv2
import numpy as np
import matplotlib.pyplot as plt

# Transformed image output size
WIDTH = 640
HEIGHT = 640
CROP = 50
# Constants for display size
DISPLAY_WIDTH = 320  # Set smaller width for display
DISPLAY_HEIGHT = 320  # Set smaller height for display

# Utility function to display images in a grid
def display_frames(original_frame, contour_frame, lines_frame, black_frame, square_frame):
    """Displays five video frames side by side with a smaller display size"""
    plt.figure(figsize=(15, 5))  # Slightly larger figure size to fit five images

    # Display original video frame
    plt.subplot(1, 5, 1)
    plt.imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
    plt.title("Original Video")
    plt.axis('off')

    # Display video with contours
    plt.subplot(1, 5, 2)
    plt.imshow(cv2.cvtColor(contour_frame, cv2.COLOR_BGR2RGB))
    plt.title("Contours")
    plt.axis('off')

    # Display video with detected lines
    plt.subplot(1, 5, 3)
    plt.imshow(cv2.cvtColor(lines_frame, cv2.COLOR_BGR2RGB))
    plt.title("Detected Lines")
    plt.axis('off')

    # Display black image with drawn lines
    plt.subplot(1, 5, 4)
    plt.imshow(cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB))
    plt.title("Black Image with Lines")
    plt.axis('off')

    # Display image with largest square contour
    plt.subplot(1, 5, 5)
    plt.imshow(cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB))
    plt.title("Largest Square Contour")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Preprocessing function with Gaussian Blur, OTSU Threshold, Canny, Hough Lines, and square contour detection
def preprocess_and_detect_lines(img):
    """Preprocess the image, detect lines, and find the largest square contour"""
    # Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    equalized_image = cv2.equalizeHist(gray_image)

    # OTSU threshold
    ret, otsu_binary = cv2.threshold(equalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny edge detection
    canny = cv2.Canny(otsu_binary, 0, 255)

    # Dilation (to connect gaps in edges)
    kernel = np.ones((7, 7), np.uint8)
    img_dilation = cv2.dilate(canny, kernel, iterations=1)

    # Hough Lines detection
    lines = cv2.HoughLinesP(img_dilation, 1, np.pi / 180, threshold=500, minLineLength=150, maxLineGap=100)

    # Create an image that contains only black pixels
    black_image = np.zeros_like(img_dilation)

    # Draw detected lines on black image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Dilation to make lines thicker and more visible
    kernel = np.ones((3, 3), np.uint8)
    black_image = cv2.dilate(black_image, kernel, iterations=1)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(black_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    square_centers = list()

    # Loop through the contours and check for square-like contours
    board_copy = canny.copy()
    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour has 4 sides (quadrilateral)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            area = cv2.contourArea(contour)

            # Only consider contours with a reasonable area range
            if 7000 < area < 40000:
                print(area)
                pts = [pt[0] for pt in approx]  # Extract coordinates
                pt1 = tuple(pts[0])
                pt2 = tuple(pts[1])
                pt4 = tuple(pts[2])
                pt3 = tuple(pts[3])

                center_x = (x + (x + w)) / 2
                center_y = (y + (y + h)) / 2

                square_centers.append([center_x, center_y, pt2, pt1, pt3, pt4])

                # Draw the square contour on the image for visualization
                cv2.line(board_copy, pt1, pt2, (255, 255, 0), 7)
                cv2.line(board_copy, pt1, pt3, (255, 255, 0), 7)
                cv2.line(board_copy, pt2, pt4, (255, 255, 0), 7)
                cv2.line(board_copy, pt3, pt4, (255, 255, 0), 7)

    return otsu_binary, canny, img_dilation, board_copy

# Process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        h, w, _ = frame.shape
        delta = (h - w) // 2
        frame = frame[delta + CROP: delta + w - CROP, :, :]

        if not ret:
            print("End of video")
            break

        # Apply preprocessing and detect lines and square contours
        otsu_binary, canny, img_dilation, black_image = preprocess_and_detect_lines(frame)

        # Display the results
        display_frames(frame, otsu_binary, canny, img_dilation, black_image)

# Path to your video file
video_path = "testing_set/2_move_student.mp4"
process_video(video_path)
