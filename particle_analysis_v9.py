import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "pencil.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Step 1: Preprocessing for Contrast and Noise Reduction
# Apply Histogram Equalization to improve contrast
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)

# Apply Bilateral Filter to reduce noise but keep edges
bilateral_filtered = cv2.bilateralFilter(equalized, d=9, sigmaColor=75, sigmaSpace=75)

# Display Preprocessed Image
plt.imshow(bilateral_filtered, cmap='gray')
plt.title("Preprocessed Image (Contrast Enhanced & Noise Reduced)")
plt.show()

# Step 2: Combined Edge Detection (Sobel + Canny)
# Sobel Edge Detection (Gradient-Based)
sobel_x = cv2.Sobel(bilateral_filtered, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(bilateral_filtered, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined = np.uint8(sobel_combined)

# Canny Edge Detection
canny_edges = cv2.Canny(bilateral_filtered, threshold1=30, threshold2=100)

# Combine Sobel and Canny
combined_edges = cv2.bitwise_or(sobel_combined, canny_edges)

# Display Combined Edges
plt.imshow(combined_edges, cmap='gray')
plt.title("Combined Edge Detection (Sobel + Canny)")
plt.show()

# Step 3: Refine Edges Using Morphological Operations
# Fill gaps using Morphological Closing
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
refined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=3)

# Save and Display Refined Edges
cv2.imwrite("refined_edges.png", refined_edges)
plt.imshow(refined_edges, cmap='gray')
plt.title("Refined Edges (Morphological Closing)")
plt.show()

# Step 4: Contour Detection and Visualization
# Detect contours from refined edges
contours, _ = cv2.findContours(refined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Save and Display Final Contour Visualization
cv2.imwrite("final_contour_visualization.png", contour_image)
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title(f"Final Contour Visualization (Shapes Detected: {len(contours)})")
plt.show()
