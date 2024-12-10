import cv2
from pathlib import Path

PATH = Path(__file__).parent
input_folder = PATH / 'images' / 'before'
output_folder = PATH / 'images' / 'after'


def modify_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        # print(f"Error: Image not found at {image_path}")
        return
    
    original_image = image.copy()

    while True:
        cv2.imshow('Image', image)
        key = cv2.waitKey(1) & 0xFF

        # Rotate CCW
        if key == ord('a') or key == 2424832:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Rotate CW
        elif key == ord('d') or key == 2555904:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # Flip
        elif key == ord('w') or key == ord('s') or key == 2490368 or key == 2621440:
            image = cv2.flip(image, 0)

        # Save
        elif key == 32 or key == 13:  
            output_path = output_folder / f"{image_path.stem}.jpg"
            cv2.imwrite(str(output_path), image)
            # print(f"Image saved as {output_path}")
            print("\033[92mImage saved.\033[0m")
            break

        # Not Save
        elif key == ord('q'):
            print("\033[91mExiting without saving.\033[0m")
            break

    cv2.destroyAllWindows()

def process_images():

    # Input
    if not input_folder.exists():
        # print(f"Input folder '{input_folder}' does not exist.")
        return

    # Output
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
        # print(f"Created output folder: {output_folder}")

    # Load images
    image_files = [f for f in input_folder.iterdir() if f.suffix == '.jpg']
    if not image_files:
        # print(f"No '.jpg' images found in '{input_folder}'.")
        return
    
    # Modify Image
    for image_file in image_files:
        print(f"Processing {image_file.name}...")
        modify_image(image_file)



if __name__ == "__main__":
    main()