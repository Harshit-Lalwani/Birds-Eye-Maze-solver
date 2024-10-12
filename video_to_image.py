import cv2
import numpy as np

def is_image_clear(image, threshold=350.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > threshold

def find_square_grid(frame, min_area, expected_size):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Adjust the blue HSV range to match the blue border around the grid
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    # Create mask for blue
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Denoise and improve the mask
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask = cv2.dilate(mask, None, iterations=1)
    mask = cv2.erode(mask, None, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            # Check if the detected contour matches the expected size of the grid
            width = np.linalg.norm(approx[0][0] - approx[1][0])
            height = np.linalg.norm(approx[1][0] - approx[2][0])
            if abs(width - expected_size) < 50 and abs(height - expected_size) < 50:
                return approx

    return None

def crop_and_perspective(frame, square_points, grid_size=1000):
    pts = square_points.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # Determine the top-left, top-right, bottom-right, and bottom-left points
    rect[0] = pts[np.argmin(np.sum(pts, axis=1))]
    rect[1] = pts[np.argmin(np.diff(pts, axis=1))]
    rect[2] = pts[np.argmax(np.sum(pts, axis=1))]
    rect[3] = pts[np.argmax(np.diff(pts, axis=1))]

    dst = np.array([[0, 0], [grid_size - 1, 0], [grid_size - 1, grid_size - 1], [0, grid_size - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (grid_size, grid_size))

    return warped

def main():
    cap = cv2.VideoCapture(0)  # Change this to a video file if needed
    frame_skip = 2
    frame_count = 0

    # Sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Adjust the size parameters (in pixels) based on camera distance
    grid_physical_size_cm = 25 * 4 + 2 * 4  # 25 squares (4cm each) + 4 cm border on both sides
    distance_to_object_cm = 50  # Approximate distance of the camera from the object
    focal_length_mm = 3.6  # Assuming standard webcam, adjust based on camera
    sensor_height_mm = 4.8  # For a typical sensor height of 4.8mm, adjust accordingly

    # Calculate expected grid size in pixels
    expected_size_pixels = int((grid_physical_size_cm / distance_to_object_cm) * (focal_length_mm / sensor_height_mm) * frame_skip * 1000)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        # Sharpen the frame to enhance details
        sharpened_frame = cv2.filter2D(frame, -1, kernel)

        if frame_count % frame_skip == 0:
            if not is_image_clear(sharpened_frame):
                print("Image is blurry, capturing next frame...")
                cv2.imshow("Square Detector", sharpened_frame)
                cv2.waitKey(100)
                frame_count += 1
                continue

            frame_area = sharpened_frame.shape[0] * sharpened_frame.shape[1]
            min_area = frame_area * 0.1  # Adjust the area to match the grid size

            # Detect the grid based on the expected size
            square_points = find_square_grid(sharpened_frame, min_area, expected_size_pixels)

            if square_points is not None:
                # Draw the detected square for debugging
                cv2.polylines(sharpened_frame, [square_points], isClosed=True, color=(0, 255, 0), thickness=2)

                # Crop and apply perspective transformation
                cropped_square = crop_and_perspective(sharpened_frame, square_points)

                # Save the cropped grid image
                cv2.imwrite('cropped_grid.png', cropped_square)
                print("Cropped grid image saved!")
                
                # Stop the video feed after finding the square
                break  # Exit the loop once the square is detected

            # Show the mask and contours for debugging
            mask = cv2.inRange(cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2HSV), np.array([90, 50, 50]), np.array([140, 255, 255]))
            cv2.imshow("Mask", mask)  # Show mask for debugging
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sharpened_frame, contours, -1, (0, 255, 0), 2)  # Draw contours for debugging
            cv2.imshow("Contours", sharpened_frame)

        cv2.imshow("Square Detector", sharpened_frame)
        frame_count += 1
        cv2.waitKey(30)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
