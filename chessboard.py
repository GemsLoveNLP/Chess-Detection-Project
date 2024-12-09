# def rotate_and_warp(frame, detection_cg):
#     # given a frame, fix it and using the new coordinate system, transform the CGs
#     return frame, detection_cg

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

class ChessboardProcessor:
    WIDTH = 640
    HEIGHT = 640

    def __init__(self, image):
        # self.image_path = image_path
        self.image = image
        self.warped_image = None
        self.transformation_matrix = None

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
        new_pts[0] = pts[np.argmin(sum_pts)]  # Top-left
        new_pts[3] = pts[np.argmax(sum_pts)]  # Bottom-right
        diff_pts = np.diff(pts, axis=1)
        new_pts[1] = pts[np.argmin(diff_pts)]  # Top-right
        new_pts[2] = pts[np.argmax(diff_pts)]  # Bottom-left
        return new_pts

    # Preprocess the image (grayscale, blur, threshold, morphology)
    def preprocess(self):
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
        img_threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 15, 2)

        # Morphological operations
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # img_opening = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, kernel)
        # img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)

        self.display_image(img_threshold, "Preprocessed Image")
        return img_threshold

    # Finds the largest contour with 4 points
    def find_biggest_contour(self, contours):
        largest_pts = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Ignore small contours
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
                if area > max_area and len(approx) == 4:
                    largest_pts = approx
                    max_area = area
        return self.reorder(largest_pts) if largest_pts is not None else None, max_area

    # Finds the chessboard corners and returns the reordered corners
    def find_board_corners(self):
        processed_img = self.preprocess()
        contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = self.image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
        self.display_image(contour_img, "Image with Contours")
        corners, _ = self.find_biggest_contour(contours)
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
        corners = self.find_board_corners()
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

            # Draw the transformed points on the warped image
            self.draw_points(warped_image, transformed_points[:, 0])

            # Display the original image with points and the warped image with transformed points
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB))
            plt.title("Original Image with Detection Points")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
            plt.title("Warped Image with Transformed Points")
            plt.axis('off')

            plt.show()

            return warped_image, transformed_points_list
        else:
            print("Could not find the corners of the chessboard.")
            return None, None



if __name__ == "__main__":
    # Example usage
    path = 'chess_model/chess_data/train/images'  # Update with your image path
    files = os.listdir(path)

    image = os.path.join(path, files[2])
    image = cv2.imread(image)

    processor = ChessboardProcessor(image)
    random_points = processor.generate_random_points() #assume these are the bbox coordinates

    transformed_image,transformed_points = processor.rotate_and_warp(image, random_points) #REPLACE random points 

    print("Transformed Points in 640x640 Plane:", transformed_points)