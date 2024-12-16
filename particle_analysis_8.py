# import cv2
# import numpy as np

# # Load the refined binary mask
# mask_image_path = "blue_mask_2.png"  # Path to your blue_mask_2.png
# mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

# # Step 1: Refine the Binary Mask
# # Define kernels for morphological operations
# kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# # Apply morphological closing to close gaps in the shapes
# refined_mask = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel_close, iterations=3)

# # Apply opening to remove noise
# refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

# # Apply dilation to strengthen shape boundaries
# refined_mask = cv2.dilate(refined_mask, kernel_close, iterations=2)

# # Save the refined mask for debugging
# cv2.imwrite("refined_mask.png", refined_mask)
# print("Refined mask saved as 'refined_mask.png'.")

# # Step 2: Detect Contours
# # Find contours in the refined mask
# contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(f"Number of shapes detected: {len(contours)}")

# # Step 3: Visualize Detected Contours
# # Create a blank canvas for visualization
# contour_visualization = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)

# # Draw contours on the canvas
# for i, contour in enumerate(contours):
#     # Draw each contour in a random color
#     color = tuple(np.random.randint(0, 255, 3).tolist())
#     cv2.drawContours(contour_visualization, [contour], -1, color, 2)

#     # Label the contours with their index
#     M = cv2.moments(contour)
#     if M["m00"] != 0:
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#         cv2.putText(contour_visualization, f"{i + 1}", (cx, cy), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# # Save and display the visualization
# cv2.imwrite("segmentation_visualization.png", contour_visualization)
# print("Segmentation visualization saved as 'segmentation_visualization.png'.")

# cv2.imshow("Refined Mask", refined_mask)
# cv2.imshow("Segmentation Visualization", contour_visualization)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import cv2
import numpy as np

# Load the binary mask generated earlier
mask_image_path = "blue_mask_2.png"  # Path to your blue_mask_2.png
mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

# Step 1: Aggressively Refine the Mask
# Define kernels for morphological operations
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # Larger kernel for closing gaps
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Kernel for removing noise

# Close gaps in shapes
refined_mask = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel_close, iterations=5)

# Remove noise using opening
refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel_open, iterations=3)
# cv2.imshow("Step 2: After Opening", refined_mask)
# cv2.imwrite("step_2_opening.png", refined_mask)
# cv2.waitKey(0)


# Dilate the mask to further strengthen boundaries
refined_mask = cv2.dilate(refined_mask, kernel_close, iterations=2)
cv2.imshow("Step 3: After Dilation", refined_mask)
cv2.imwrite("step_3_dilation.png", refined_mask)
cv2.waitKey(0)



# Step 2: Fill Internal Gaps (Flood Fill)
# h, w = refined_mask.shape[:2]
# flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
# flood_filled = refined_mask.copy()
# cv2.floodFill(flood_filled, flood_fill_mask, (0, 0), 255)  # Fill background regions
# inverse_filled = cv2.bitwise_not(flood_filled)  # Invert to get internal holes
# filled_mask = cv2.bitwise_or(refined_mask, inverse_filled)  # Combine filled regions with the mask

# cv2.imshow("Step 4: After Filling Gaps", filled_mask)
# cv2.imwrite("step_4_filled_gaps.png", filled_mask)
# cv2.waitKey(0)



# # Save the refined mask for verification
# cv2.imwrite("final_refined_mask.png", filled_mask)
# print("Final refined mask saved as 'final_refined_mask.png'.")

# # Step 3: Detect Contours with Canny Edge Detection
# # Apply Canny edge detection
# edges = cv2.Canny(filled_mask, 50, 150)

# # Detect contours from edges
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(f"Number of shapes detected: {len(contours)}")

# # Step 4: Visualize the Detected Contours
# # Create a blank canvas for visualization
# visualization = cv2.cvtColor(filled_mask, cv2.COLOR_GRAY2BGR)

# # Draw contours on the canvas
# for i, contour in enumerate(contours):
#     # Generate a random color for each contour
#     color = tuple(np.random.randint(0, 255, 3).tolist())
#     cv2.drawContours(visualization, [contour], -1, color, 2)

#     # Label the contours with their index
#     M = cv2.moments(contour)
#     if M["m00"] != 0:
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#         cv2.putText(visualization, f"{i + 1}", (cx, cy),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# # Save the visualization
# cv2.imwrite("improved_segmentation_visualization.png", visualization)
# print("Segmentation visualization saved as 'improved_segmentation_visualization.png'.")

# # Display the final mask and visualization
# cv2.imshow("Final Refined Mask", filled_mask)
# cv2.imshow("Improved Segmentation Visualization", visualization)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
