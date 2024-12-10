import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

class ChessboardProcessor:
    WIDTH = 640 # Fixed Height of output (grids will be 80x80 px)
    HEIGHT = 640

    # Define max and min contour areas for corner detection
    MAX_CONTOUR_AREA = 40000 
    MIN_CONTOUR_AREA = 7000

    last_warped_image = None

    def __init__(self, image):
        self.image = image
        self.warped_image = None
        self.homo_matrix = None

    # Utility function to display images
    def display_image(self, img, title="Image"):
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(title)
        plt.show()

    # Reorders points to top-left, top-right, bottom-right, bottom-left
    def reorder(self, pts):
        pts = pts.reshape((4, 2))
        new_pts = np.zeros((4, 1, 2), dtype="float32")
        sum_pts = pts.sum(1)
        diff_pts = np.diff(pts, axis=1)

        new_pts[0] = pts[np.argmin(sum_pts)]  # Top-left
        new_pts[3] = pts[np.argmax(sum_pts)]  # Bottom-right
        
        new_pts[1] = pts[np.argmin(diff_pts)]  # Top-right
        new_pts[2] = pts[np.argmax(diff_pts)]  # Bottom-left

        return new_pts

    def find_largest_contour(self, max_board):
        # Dilate the image to connect inner square lines 
        kernel = np.ones((5, 5), np.uint8)
        dilated_max_board = cv2.dilate(max_board, kernel, iterations=1)

        # Find the contours of dilated image
        outer_contours, _ = cv2.findContours(dilated_max_board, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_pts = None 

        # Find contour polygon of largest area (assume to be chessboard)
        max_area = 0
        for contour in outer_contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Ignore small contours
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    largest_pts = approx
                    max_area = area

        return self.reorder(largest_pts) if largest_pts is not None else None

    # Preprocessing to obtain larger square
    def find_corners(self):
        # Grayscale and Equalize (counter brightness issues)
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)

        # Otsu's thresholding
        ret, otsu_binary = cv2.threshold(equalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Canny edgea
        canny = cv2.Canny(otsu_binary, 0, 255)

        # Dilation (to connect gaps in edges)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        img_dilation = cv2.dilate(canny, kernel, iterations=1)

        # Hough Lines detection
        lines = cv2.HoughLinesP(img_dilation, 1, np.pi / 180, threshold=500, minLineLength=150, maxLineGap=100)

        black_image = np.zeros_like(img_dilation) # Canvas for drawing dilated lines
        max_board = np.zeros_like(img_dilation) # Canvas for drawing outer chessboard edges later

        # Draw resulting lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Dilation to make lines thicker and more visible
        kernel = np.ones((3, 3), np.uint8)
        black_image = cv2.dilate(black_image, kernel, iterations=1) # Returns a black image with lines representing grids

        # Find contours in the dilated image
        contours, _ = cv2.findContours(black_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        result_image = self.image.copy()  # Use original image as base for drawing contours

        # Loop through contours to find inner squares
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True) 
            approx = cv2.approxPolyDP(contour, epsilon, True) # Approximate contours representing polygons

            if len(approx) == 4:  # Only approx 4 sides
                (x, y, w, h) = cv2.boundingRect(approx)
                area = cv2.contourArea(contour)
                
                if self.MIN_CONTOUR_AREA < area < self.MAX_CONTOUR_AREA:
                    aspect_ratio = w / float(h)
                    
                    # Decrease chances of finding rectangles instead
                    if (0.2 <= aspect_ratio <= 1.3) and (len(approx) == 4): 
                        pts = [tuple(pt[0]) for pt in approx]
                        pt1, pt2, pt3, pt4 = pts

                        # Draw the square contour on the image for visualization (RED for inner squares)
                        cv2.line(result_image, pt1, pt2, (0, 0, 255), 7)  # Red
                        cv2.line(result_image, pt1, pt3, (0, 0, 255), 7)  # Red
                        cv2.line(result_image, pt2, pt4, (0, 0, 255), 7)  # Red
                        cv2.line(result_image, pt3, pt4, (0, 0, 255), 7)  # Red

                        cv2.line(max_board, pt1, pt2, (255,255,255), 7)  # Copy lines into separate board to find outer square
                        cv2.line(max_board, pt1, pt3, (255,255,255), 7)  
                        cv2.line(max_board, pt2, pt4, (255,255,255), 7)  
                        cv2.line(max_board, pt3, pt4, (255,255,255), 7)  

        corners = self.find_largest_contour(max_board)
        return corners
    
    # Applies perspective transformation to get a bird's-eye view
    def warp_image(self, corners):
        original_pts = np.float32(corners)
        new_pts = np.float32([[0, 0], [self.WIDTH, 0], [0, self.HEIGHT], [self.WIDTH, self.HEIGHT]])
        self.transformation_matrix = cv2.getPerspectiveTransform(original_pts, new_pts)
        self.warped_image = cv2.warpPerspective(self.image, self.transformation_matrix, (self.WIDTH, self.HEIGHT))
        return self.warped_image

    # Draws points on the given image
    def draw_points(self, img, pts, color=(0, 0, 255), size=10):
        for pt in pts:
            pt = tuple(map(int, pt))  # Convert to (x, y) tuple
            cv2.circle(img, pt, size, color, -1)

    # Generates random points within the image dimensions
    def generate_random_points(self, num_points=4):
        height, width, _ = self.image.shape
        
        # Generate random points
        random_points = [(random.randint(0, width - 1), random.randint(0, height - 1)) for _ in range(num_points)]
        random_points = np.float32(random_points)
        
        # Create a list of tuples with (piece, x_coord, y_coord)
        piece_list = []
        for i, (x_coord, y_coord) in enumerate(random_points, start=1):
            piece_list.append((f"piece {i}", x_coord, y_coord))
        
        return piece_list


    # Main function to process the image
    def rotate_and_warp(self, detection_cg):
        # Find the chessboard corners in the frame
        corners = self.find_corners()

        if corners is not None:
            # Copy the original image for visualization
            image_with_points = self.image.copy()

            # Draw the original detection CG points on the original image
            for piece, x_coord, y_coord in detection_cg:
                self.draw_points(image_with_points, [(x_coord, y_coord)])

            # Warp the image based on the detected chessboard corners
            warped_image = self.warp_image(corners)

            # Prepare the original detection CG points for transformation
            original_points = np.array([[x, y] for _, x, y in detection_cg], dtype=np.float32).reshape(-1, 1, 2)

            # Apply perspective transformation to the original points
            transformed_points = cv2.perspectiveTransform(original_points, self.transformation_matrix)

            # Prepare the transformed points as a list of tuples (piece, x_new, y_new)
            transformed_points_list = []
            for i, (piece, _, _) in enumerate(detection_cg):  # Correct unpacking here
                x_new, y_new = transformed_points[i][0]
                transformed_points_list.append((piece, x_new, y_new))

            # Store the last valid warped image
            self.last_warped_image = warped_image

            # Draw the transformed points on the warped image
            # self.draw_points(warped_image, transformed_points[:, 0])

            return warped_image, transformed_points_list
        else:
            # print("Could not find the corners of the chessboard.")
            # If no corners are detected, use the last valid warped image
            if self.last_warped_image is not None:
                # print("Using the last valid warped image.")
                return self.last_warped_image, []  # Return the last warped image and empty points
            else:
                # print("No previous valid warped image available.")
                return None, None

# Main function
def main():
    video_path = "testing_set/2_move_student.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    CROP = 50  # Define the crop value used to adjust frames, adjust based on your needs

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video")
            break

        h, w, _ = frame.shape
        delta = (h - w) // 2
        frame = frame[delta + CROP: delta + w - CROP, :, :]

        processor = ChessboardProcessor(frame)
        detection_cg = processor.generate_random_points()

        transformed_image, transformed_points = processor.rotate_and_warp(frame, detection_cg)

        # Resize frames to display them smaller
        resized_frame = cv2.resize(frame, (640, 640))

        # Show original and transformed frames
        cv2.imshow("Original Video", resized_frame)

        # If a transformed image is available, show it
        if transformed_image is not None:
            cv2.imshow("Transformed Video", transformed_image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()