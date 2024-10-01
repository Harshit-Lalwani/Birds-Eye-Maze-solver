import cv2
import numpy as np

def is_image_clear(image, threshold=350.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > threshold

def find_blue_square(frame, min_area):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Adjusted to include a wider range of blue
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Optional: Denoise the mask to remove small noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            return approx

    return None

def crop_and_perspective(frame, square_points):
    # Define the points for the perspective transform
    pts = square_points.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left point
    rect[0] = pts[np.argmin(np.sum(pts, axis=1))]
    # Top-right point
    rect[1] = pts[np.argmin(np.diff(pts, axis=1))]
    # Bottom-right point
    rect[2] = pts[np.argmax(np.sum(pts, axis=1))]
    # Bottom-left point
    rect[3] = pts[np.argmax(np.diff(pts, axis=1))]

    # Calculate the width and height of the new image
    widthA = np.linalg.norm(rect[0] - rect[1])
    widthB = np.linalg.norm(rect[2] - rect[3])
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(rect[0] - rect[3])
    heightB = np.linalg.norm(rect[1] - rect[2])
    maxHeight = max(int(heightA), int(heightB))

    # Set the destination points for the perspective transform
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

    return warped

def main():
    cap = cv2.VideoCapture(0)
    frame_skip = 3
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        if frame_count % frame_skip == 0:
            if not is_image_clear(frame):
                print("Image is blurry, capturing next frame...")
                cv2.imshow("Square Detector", frame)
                cv2.waitKey(100)
                frame_count += 1
                continue

            frame_area = frame.shape[0] * frame.shape[1]
            min_area = frame_area * 0.1
            square_points = find_blue_square(frame, min_area)

            if square_points is not None:
                # Draw the detected square (optional)
                cv2.polylines(frame, [square_points], isClosed=True, color=(0, 255, 0), thickness=2)

                # Crop the square with perspective transform
                cropped_square = crop_and_perspective(frame, square_points)

                # Save the cropped square image
                cv2.imwrite('cropped_inner_square.png', cropped_square)
                print("Cropped inner square image saved!")
                
                # Stop the video feed after finding the square
                cap.release()
                cv2.destroyAllWindows()
                return

        cv2.imshow("Square Detector", frame)
        frame_count += 1
        cv2.waitKey(30)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

