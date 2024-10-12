import cv2
import numpy as np
import os

# Function to find objects with blue borders in the image
def find_blue_border_object(frame, min_area):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_object = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and area > max_area:
            largest_object = approx
            max_area = area

    return largest_object

# Function to find objects with outer blue borders in the image
def find_outer_blue_border_object(frame, min_area):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_object = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if area > max_area:
            largest_object = approx
            max_area = area

    return largest_object

# Function to wrap the image from its corners into a square
def wrap_to_square(image, multiple_of=25):
    height, width, _ = image.shape

    # Define the original corner points (assuming the outer blue border)
    src_points = np.float32([
        [0, 0],               # Top-left corner
        [width - 1, 0],        # Top-right corner
        [width - 1, height - 1],  # Bottom-right corner
        [0, height - 1]        # Bottom-left corner
    ])

    # Define the destination points (for a perfect square)
    side_length = max(height, width)
    dst_points = np.float32([
        [0, 0],                     # Top-left corner
        [side_length - 1, 0],        # Top-right corner
        [side_length - 1, side_length - 1],  # Bottom-right corner
        [0, side_length - 1]         # Bottom-left corner
    ])

    # Perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Warp the image to make it a perfect square
    wrapped_image = cv2.warpPerspective(image, matrix, (side_length, side_length))

    # Ensure the size is a multiple of 4
    if side_length % multiple_of != 0:
        final_size = side_length + (multiple_of - side_length % multiple_of)
        wrapped_image = cv2.resize(wrapped_image, (final_size, final_size))

    return wrapped_image

# Step 1: Capture video, detect blue-border object, and save the cropped image
def detect_and_crop_blue_border_object():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * 0.05
        blue_border_object = find_blue_border_object(frame, min_area)

        if blue_border_object is not None:
            x, y, w, h = cv2.boundingRect(blue_border_object)
            cropped_image = frame[y:y+h, x:x+w]
            output_image_path = os.path.join(os.getcwd(), 'cropped_blue_border_object.png')
            cv2.imwrite(output_image_path, cropped_image)
            print(f"Cropped image saved at: {output_image_path}")
            cap.release()
            cv2.destroyAllWindows()
            return output_image_path  # Return the path to the cropped image
        else:
            print("No blue-border object detected in this frame.")
        
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

# Step 2: Use the cropped image as input and detect outer blue border, then wrap to square
def detect_outer_blue_border_and_wrap(input_image_path):
    frame = cv2.imread(input_image_path)
    if frame is None:
        print(f"Failed to load image: {input_image_path}")
        return

    frame_area = frame.shape[0] * frame.shape[1]
    min_area = frame_area * 0.05
    blue_border_object = find_outer_blue_border_object(frame, min_area)

    if blue_border_object is not None:
        if len(blue_border_object) != 4:
            epsilon = 0.05 * cv2.arcLength(blue_border_object, True)
            blue_border_object = cv2.approxPolyDP(blue_border_object, epsilon, True)

        if len(blue_border_object) == 4:
            x, y, w, h = cv2.boundingRect(blue_border_object)
            src_points = np.float32(blue_border_object.reshape(-1, 2))
            dst_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])

            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            warped_image = cv2.warpPerspective(frame, matrix, (w, h))

            # Save warped image
            output_image_path = os.path.join(os.getcwd(), 'wrapped_outer_blue_border.png')
            cv2.imwrite(output_image_path, warped_image)
            print(f"Wrapped outer blue border image saved at: {output_image_path}")

            # Now wrap the image into a square
            square_image = wrap_to_square(warped_image, multiple_of=4)
            square_output_image_path = os.path.join(os.getcwd(), 'wrapped_square_image.png')
            cv2.imwrite(square_output_image_path, square_image)
            print(f"Wrapped square image saved at: {square_output_image_path}")
        else:
            print("Could not approximate to exactly 4 points.")
    else:
        print("No outer blue-border object detected in the image.")

def main():
    # Step 1: Detect and crop blue border object
    cropped_image_path = detect_and_crop_blue_border_object()

    if cropped_image_path:
        # Step 2: Detect outer blue border and warp to square using the cropped image
        detect_outer_blue_border_and_wrap(cropped_image_path)

if __name__ == "__main__":
    main()
