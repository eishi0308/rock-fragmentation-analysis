import cv2
import numpy as np
import random

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

# Step 2: Refine the Binary Mask
# Use morphological operations to close gaps
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Small rectangular kernel
refined_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

# Fill holes in the mask
refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)  # Dilation to thicken lines
refined_mask = cv2.erode(refined_mask, kernel, iterations=1)   # Erosion to clean artifacts

# Save the refined mask for verification
cv2.imwrite("refined_blue_mask.png", refined_mask)
print("Refined mask saved as 'refined_blue_mask.png'.")

# Step 3: Detect Contours
# Find the contours in the refined binary mask
contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Number of shapes detected: {len(contours)}")


###############################################################################################################################

# Step 4: Label Each Detected Shape
# Create a copy of the original image to draw labels
labeled_image = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)  # Convert mask to BGR for labeling

for i, contour in enumerate(contours):
    # Get the center of each contour to place the label
    M = cv2.moments(contour)
    if M["m00"] != 0:  # Avoid division by zero
        cx = int(M["m10"] / M["m00"])  # X coordinate of the center
        cy = int(M["m01"] / M["m00"])  # Y coordinate of the center
        # Label the shape with its number
        cv2.putText(labeled_image, str(i + 1), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    else:
        # Handle case where the contour has zero area (unlikely in this case)
        continue

# Save the labeled image
cv2.imwrite("labeled_shapes.png", labeled_image)
print("Labeled image saved as 'labeled_shapes.png'.")





# # Create a blank canvas for visualization
# single_color_canvas = np.zeros((refined_mask.shape[0], refined_mask.shape[1], 3), dtype=np.uint8)

# # Define a single color for filling
# fill_color = (0, 255, 0)  # Green color

# # Fill each detected shape with the single color
# for contour in contours:
#     # Fill the contour
#     cv2.drawContours(single_color_canvas, [contour], -1, fill_color, thickness=cv2.FILLED)

# # Overlay the original contours as white outlines for visibility
# for contour in contours:
#     cv2.drawContours(single_color_canvas, [contour], -1, (255, 255, 255), thickness=1)  # White outlines

# # Save the single-color visualization
# cv2.imwrite("single_color_shapes.png", single_color_canvas)
# print("Single-color visualization saved as 'single_color_shapes.png'.")





# # Create a blank canvas to color the detected shapes
# colored_shapes = np.zeros((refined_mask.shape[0], refined_mask.shape[1], 3), dtype=np.uint8)

# # Assign random colors to each shape
# for i, contour in enumerate(contours):
#     # Generate a random color
#     color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

#     # Fill the contour with the random color
#     cv2.drawContours(colored_shapes, [contour], -1, color, thickness=cv2.FILLED)

# # Step 5: Overlay the Contours on the Colored Image
# # Draw the original white contours for better alignment
# for contour in contours:
#     cv2.drawContours(colored_shapes, [contour], -1, (255, 255, 255), thickness=1)

# # Save the improved colored visualization
# cv2.imwrite("improved_colored_shapes.png", colored_shapes)
# print("Improved colored visualization saved as 'improved_colored_shapes.png'.")


# # Fill each detected shape with the single color
# for contour in contours:
#     # Fill the contour
#     cv2.drawContours(single_color_canvas, [contour], -1, fill_color, thickness=cv2.FILLED)

# # Overlay the original contours as white outlines for visibility
# for contour in contours:
#     cv2.drawContours(single_color_canvas, [contour], -1, (255, 255, 255), thickness=1)  # White outlines

# # Save the single-color visualization
# cv2.imwrite("single_color_shapes.png", single_color_canvas)
# print("Single-color visualization saved as 'single_color_shapes.png'.")






# # Save the colored visualization
# cv2.imwrite("colored_shapes.png", colored_shapes)
# print("Colored visualization saved as 'colored_shapes.png'.")

# # Display the colored shapes
# cv2.imshow("Colored Shapes", colored_shapes)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# # Step 4: Annotate Detected Shapes
# # Create an annotated image
# annotated_image = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)
# for i, contour in enumerate(contours):
#     # Draw the contour on the image
#     cv2.drawContours(annotated_image, [contour], -1, (0, 255, 0), 1)

#     # Get the bounding box for the contour
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.putText(annotated_image, f"{i + 1}", (x, y - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# # Save and display the annotated image
# cv2.imwrite("annotated_shapes.png", annotated_image)
# print("Annotated shapes saved as 'annotated_shapes.png'.")
# cv2.imshow("Annotated Shapes", annotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
