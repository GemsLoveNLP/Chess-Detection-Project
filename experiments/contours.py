import cv2
import numpy as np
import matplotlib.pyplot as plt

# Transformed image output size
WIDTH = 640
HEIGHT = 640
CROP = 50

# Resizing factor (you can adjust this value to make the frames smaller or larger)
RESIZE_FACTOR = 0.5  # 50% size of the original frames

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
    
    # Loop through the contours and check for square-like contours

    max_board = np.zeros_like(img_dilation)
    board_copy = img.copy()  # Use original image as base for drawing contours

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
                # Check aspect ratio for squareness (w should be equal to h for square)
                aspect_ratio = w / float(h)
                
                # Condition for square (aspect ratio close to 1)
                if 0.2 <= aspect_ratio <= 1.3:  # Tolerance for slight deviations from perfect square
                    pts = [pt[0] for pt in approx]  # Extract coordinates
                    pt1 = tuple(pts[0])
                    pt2 = tuple(pts[1])
                    pt4 = tuple(pts[2])
                    pt3 = tuple(pts[3])

                    # Draw the square contour on the image for visualization (RED for inner squares)
                    cv2.line(board_copy, pt1, pt2, (0, 0, 255), 7)  # Red
                    cv2.line(board_copy, pt1, pt3, (0, 0, 255), 7)  # Red
                    cv2.line(board_copy, pt2, pt4, (0, 0, 255), 7)  # Red
                    cv2.line(board_copy, pt3, pt4, (0, 0, 255), 7)  # Red

                    cv2.line(max_board, pt1, pt2, (255,255,255), 7)  # Copy lines into separate board to find outer square
                    cv2.line(max_board, pt1, pt3, (255,255,255), 7)  
                    cv2.line(max_board, pt2, pt4, (255,255,255), 7)  
                    cv2.line(max_board, pt3, pt4, (255,255,255), 7)  
        
    kernel = np.ones((5, 5), np.uint8)
    dilated_max_board = cv2.dilate(max_board, kernel, iterations=1)
    black_board = np.zeros_like(img_dilation)
    outer_contours, _ = cv2.findContours(dilated_max_board, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_pts = None
    max_area = 0
    for contour in outer_contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Ignore small contours
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                largest_pts = approx
                max_area = area
                pts = largest_pts
                pt1 = tuple(pts[0][0])
                pt2 = tuple(pts[1][0])
                pt4 = tuple(pts[2][0])
                pt3 = tuple(pts[3][0])
    cv2.line(board_copy, pt1, pt2, (0,255,0), 7)  # Copy lines into separate board to find outer square
    cv2.line(board_copy, pt1, pt3, (0,255,0), 7)  
    cv2.line(board_copy, pt2, pt4, (0,255,0), 7)  
    cv2.line(board_copy, pt3, pt4, (0,255,0), 7)  

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
        otsu_binary, canny, img_dilation, board_copy = preprocess_and_detect_lines(frame)

        # Resize frames to make them smaller
        resized_frame = cv2.resize(frame, (500,500))
        resized_board_copy = cv2.resize(board_copy, (500,500))

        # Show original frame in a resized window
        cv2.imshow("Original Video", resized_frame)

        # Show the final output (board_copy) in a resized window
        cv2.imshow("Transformed Video", resized_board_copy)

        # Wait for a key press and break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Path to your video file
video_path = "testing_set/2_move_student.mp4"
process_video(video_path)
