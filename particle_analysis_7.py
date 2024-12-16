import cv2
import numpy as np

# Load the image
image_path = "sample_3.png"  # Replace with the actual image path
image = cv2.imread(image_path)

# Step 1: Convert to HSV to Isolate Blue Lines
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the HSV range for blue
lower_blue = np.array([100, 150, 50])  # Adjust if needed
upper_blue = np.array([140, 255, 255])

# Create a binary mask for blue lines
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Step 2: Enhance the Binary Mask
# Use morphological operations to close gaps and strengthen lines
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Small rectangular kernel

# Apply closing to fill gaps in the lines
blue_mask_refined = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

# Apply opening to remove noise and small artifacts
blue_mask_refined = cv2.morphologyEx(blue_mask_refined, cv2.MORPH_OPEN, kernel, iterations=2)

# Apply dilation to further strengthen lines
blue_mask_refined = cv2.dilate(blue_mask_refined, kernel, iterations=1)

# Save the refined mask for verification
cv2.imwrite("refined_blue_mask_2.png", blue_mask_refined)
print("Refined blue mask saved as 'refined_blue_mask_2.png'.")

# Step 3: Detect Contours with Improved Refinement
# Find contours in the refined mask
contours, _ = cv2.findContours(blue_mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Number of shapes detected after refinement: {len(contours)}")

# Step 4: Visualize Contours
# Create a copy of the refined mask to visualize contours
contour_visualization = cv2.cvtColor(blue_mask_refined, cv2.COLOR_GRAY2BGR)

# Draw all contours in green
cv2.drawContours(contour_visualization, contours, -1, (0, 255, 0), 1)

# Save the visualization
cv2.imwrite("contours_visualization.png", contour_visualization)
print("Contour visualization saved as 'contours_visualization.png'.")

# Display the refined mask and contour visualization for debugging
cv2.imshow("Refined Mask", blue_mask_refined)
cv2.imshow("Contours Visualization", contour_visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()
