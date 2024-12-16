import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Save the blue mask as a new file
cv2.imwrite("blue_mask_2.png", blue_mask)
print("Blue mask saved as 'blue_mask_2.png'.")


# Load the mask image 
mask_image_path = "blue_mask_2.png"  # Path to your blue_mask_2.png
mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale


# Step : Detect Contourss
# Find the contours in the binary mask
contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Number of shapes detected: {len(contours)}")


######################################################################################################################



# Step 2: Morphological operations to clean the mask
kernel = np.ones((3, 3), np.uint8)  # Define a kernel
cleaned_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)  # Close gaps
cv2.imwrite("blue_mask_cleaned.png", cleaned_mask)

# Step 3: Apply Gaussian blur for smoothing
smoothed_mask = cv2.GaussianBlur(cleaned_mask, (5, 5), 0)
cv2.imwrite("blue_mask_smoothed.png", smoothed_mask)

# Step 4: Detect contours
contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(f"Number of shapes detected: {len(contours)}")

# Draw contours for visualization
contours_image = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 1)
cv2.imwrite("contours_detected.png", contours_image)

# Step 5: Canny Edge Detection
edges = cv2.Canny(cleaned_mask, 100, 200)
cv2.imwrite("canny_edges.png", edges)

# Step 6: Hough Line Transformation
hough_lines_image = blue_mask.copy()
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(hough_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite("hough_lines_detected.png", hough_lines_image)

# Step 7: Invert the binary mask to use for the watershed algorithm
inverted_mask = cv2.bitwise_not(cleaned_mask)
cv2.imwrite("inverted_mask.png", inverted_mask)

# Create markers for watershed
_, markers = cv2.connectedComponents(inverted_mask)

# Convert markers to the required type
markers = markers.astype(np.int32)

# Convert the mask to a 3-channel image for watershed
watershed_image = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)

# Apply the watershed algorithm
cv2.watershed(watershed_image, markers)

# Mark boundaries in green
watershed_image[markers == -1] = [0, 255, 0]  # Mark boundaries with green

cv2.imwrite("watershed_result.png", watershed_image)

# Visualization of intermediate results
# steps = [
#     ("Original Blue Mask", "blue_mask.png"),
#     ("Cleaned Mask (Morphological Operations)", "blue_mask_cleaned.png"),
#     ("Smoothed Mask (Gaussian Blur)", "blue_mask_smoothed.png"),
#     ("Contours Detected", "contours_detected.png"),
#     ("Canny Edges", "canny_edges.png"),
#     ("Hough Lines Detected", "hough_lines_detected.png"),
#     ("Inverted Mask", "inverted_mask.png"),
#     ("Watershed Result", "watershed_result.png"),
# ]

# for title, path in steps:
#     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
#     plt.title(title)
#     plt.axis("off")
#     plt.show()




# Load the watershed result image
watershed_image_path = "watershed_result.png"
watershed_image = cv2.imread(watershed_image_path)


# Assume the watershed markers are available as `markers` from the watershed process
# If not, you would need to recompute the markers directly
# Use markers to detect unique shapes
unique_markers = np.unique(markers)

# Exclude background (0) and boundary (-1)
shape_markers = [marker for marker in unique_markers if marker > 0]
shape_count = len(shape_markers)

print(f"Number of shapes detected: {shape_count}")


########################################
#shape counting and visualization
########################################

# Visualize detected shapes
annotated_image = cv2.cvtColor(watershed_image, cv2.COLOR_BGR2RGB)

# Annotate shape IDs
for marker in shape_markers:
    # Find the centroid of each shape
    mask = (markers == marker).astype(np.uint8)
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        cv2.putText(
            annotated_image,
            f"{marker}",
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

# Display annotated image
plt.figure(figsize=(10, 10))
plt.imshow(annotated_image)
plt.title(f"Detected Shapes: {shape_count}")
plt.axis("off")
plt.show()




################################################################
################################################################


#To calculate the size of each shape (area) in the segmented image


import numpy as np
import cv2

# Load the markers array (from the watershed process)
# Ensure `markers` is already computed in your watershed process
# markers = ...

# Initialize a dictionary to store the area of each shape
shape_areas = {}

# Exclude background (0) and boundary (-1)
unique_markers = np.unique(markers)
print(f"Unique markers (including background and boundary): {unique_markers}")

# Iterate over each unique marker
for marker in unique_markers:
    if marker <= 0:  # Skip background and boundary
        continue
    
    # Count pixels belonging to the current marker
    area = np.sum(markers == marker)  # Count where the marker equals the current value
    shape_areas[marker] = area

# Print the results
for marker, area in shape_areas.items():
    print(f"Shape ID {marker}: Area = {area} pixels")


################################################################
##CSV file
################################################################

import csv

# Save the results to a CSV file
with open("shape_areas.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Shape ID", "Area (pixels)"])
    for marker, area in shape_areas.items():
        writer.writerow([marker, area])

print("Shape areas saved to 'shape_areas.csv'.")


################
# Visualization
################

# import matplotlib.pyplot as plt

# # Create a copy of the watershed image for annotation
# annotated_image = cv2.cvtColor(watershed_image, cv2.COLOR_BGR2RGB)

# # Annotate each shape with its area
# for marker, area in shape_areas.items():
#     # Find the centroid of the shape for annotation
#     mask = (markers == marker).astype(np.uint8)
#     moments = cv2.moments(mask)
#     if moments["m00"] != 0:
#         cx = int(moments["m10"] / moments["m00"])
#         cy = int(moments["m01"] / moments["m00"])
#         cv2.putText(
#             annotated_image,
#             f"{area}",
#             (cx, cy),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (255, 0, 0),
#             1,
#         )

# # Display the annotated image
# plt.figure(figsize=(10, 10))
# plt.imshow(annotated_image)
# plt.title("Shape Areas Annotated")
# plt.axis("off")
# plt.show()


######################################################################################################################
######################################################################


########################################
# Shape Area Calculation
########################################

# Initialize a dictionary to store the area of each shape
shape_areas = {}

# Iterate over each unique marker
for marker in shape_markers:
    # Count pixels belonging to the current marker
    area = np.sum(markers == marker)  # Count where the marker equals the current value
    shape_areas[marker] = area

# Print the areas in pixels
print("\nShape Areas in Pixels:")
for marker, area in shape_areas.items():
    print(f"Shape ID {marker}: Area = {area} pixels")


########################################
# Real-World Area Calculation
########################################

# Step 1: Ask the user for resolution (pixels per meter)
try:
    px_per_meter = float(input("Enter the image resolution in pixels per meter (px/m): "))
except ValueError:
    print("Invalid input! Please enter a numeric value.")
    exit()

# Step 2: Convert pixel areas to real-world areas
real_shape_areas = {}
conversion_factor = 1 / px_per_meter  # Convert px to meters
for marker, area_pixels in shape_areas.items():
    area_real = area_pixels * (conversion_factor ** 2)  # Convert to m²
    real_shape_areas[marker] = area_real

# Print the areas in real-world units
print("\nShape Areas in Real-World Units:")
for marker, area_real in real_shape_areas.items():
    print(f"Shape ID {marker}: Real Area = {area_real:.6f} m²")



######################################################################################################################
######################################################################


# Display the blue mask for debugging
# cv2.imshow("Blue Mask", blue_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Debugging: Save and check blue mask
# cv2.imwrite("blue_mask_debug.png", blue_mask)
# print("Blue mask created. Verifying non-zero pixels...")
# print(f"Non-zero pixels in blue mask: {np.count_nonzero(blue_mask)}")

# # Debugging: Show the mask
# cv2.imshow("Blue Mask", blue_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # Step 2: Refine the Blue Mask
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# refined_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

# # Debugging: Save and check refined mask
# cv2.imwrite("refined_mask_debug.png", refined_mask)
# print("Refined mask created. Verifying non-zero pixels...")
# print(f"Non-zero pixels in refined mask: {np.count_nonzero(refined_mask)}")

# # Display the refined mask for debugging
# cv2.imshow("Refined Mask", refined_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Step 3: Detect Contours
# contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(f"Number of contours detected: {len(contours)}")  # Debugging print

# # Check if any contours are detected
# if len(contours) == 0:
#     print("No contours detected. Check your mask or preprocessing steps.")
# else:
#     print(f"Number of contours detected: {len(contours)}")



# # Step 4: Draw Contours on the Image
# contour_image = image.copy()
# cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Green contours

# # Display the contours on the image
# cv2.imshow("Contours", contour_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Save the contours image
# cv2.imwrite("contours_with_green_lines.png", contour_image)
# print("Contours image saved as 'contours_with_green_lines.png'")

# # Step 5: Define Scale and Calculate Areas
# scale_length_in_pixels = 500  # Replace with the actual pixel length of the scale in your image
# real_scale_length_cm = 100  # Known real-world length of the scale (e.g., 100 cm)
# pixels_per_cm = scale_length_in_pixels / real_scale_length_cm

# particle_areas = []
# for contour in contours:
#     # Calculate the area in pixels
#     pixel_area = cv2.contourArea(contour)
#     # Convert to real-world area in cm²
#     real_area = pixel_area / (pixels_per_cm ** 2)
#     particle_areas.append(real_area)

# # Print the particle areas
# for i, area in enumerate(particle_areas):
#     print(f"Particle {i + 1}: {area:.2f} cm²")




# # Step 6: Annotate Particle Sizes on the Image
# for i, contour in enumerate(contours):
#     x, y, w, h = cv2.boundingRect(contour)
#     annotation = f"{particle_areas[i]:.2f} cm²"

#     # Draw a black rectangle for the text background
#     font_scale = 0.5
#     font_thickness = 1
#     text_size = cv2.getTextSize(annotation, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
#     text_width, text_height = text_size
#     cv2.rectangle(
#         contour_image,
#         (x, y - text_height - 5),  # Top-left corner of the rectangle
#         (x + text_width, y),  # Bottom-right corner
#         (0, 0, 0),  # Black rectangle
#         -1  # Filled rectangle
#     )

#     # Draw the annotation text
#     cv2.putText(
#         contour_image,
#         annotation,
#         (x, y - 5),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         font_scale,
#         (255, 255, 255),  # White text
#         font_thickness
#     )

# # Save and display the annotated image
# cv2.imwrite("annotated_particles_with_sizes.png", contour_image)
# cv2.imshow("Annotated Image", contour_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
