import cv2
import numpy as np

def detect_palm_lines(image_path, output_path='static/uploads/lines_detected.jpg'):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return "Image not found or invalid path."

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Sharpen image to highlight palm lines
    kernel_sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(blur, -1, kernel_sharpen)

    # Canny edge detection - lower thresholds for finer line detection
    edges = cv2.Canny(sharpened, 20, 80)

    # Dilate the edges to enhance
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours in the dilated edge map
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Remove largest contour (likely hand outer boundary), only if there are more than 2
    if len(contours) > 1:
        contours = contours[1:]

    # Draw remaining contours onto a copy of the original image
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

    # Save the result
    cv2.imwrite(output_path, contour_img)
    return output_path

# Example usage for standalone testing
if __name__ == "__main__":
    input_image = 'static/uploads/sample_palm.jpg'  # Replace with your sample file
    output_image = detect_palm_lines(input_image)
    print("Output saved to:", output_image)
