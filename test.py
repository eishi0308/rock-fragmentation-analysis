# import cv2
# import numpy as np

# # Load the image
# image_path = "sample_3.png"  # Replace with your actual image path
# image = cv2.imread(image_path)

# # Step 1: Convert the Image to Grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Step 2: Threshold to Create a Binary Image
# # Use the binary mask (white lines on black background)
# _, binary_mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# # Step 3: Detect Contours
# contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(f"Number of contours detected: {len(contours)}")

# # Step 4: Define Scale for Real-World Conversion
# scale_length_in_pixels = 500  # Replace with your scale length in pixels
# real_scale_length_cm = 100  # Known real-world length in cm
# pixels_per_cm = scale_length_in_pixels / real_scale_length_cm

# # Step 5: Calculate Areas and Annotate
# particle_areas = []
# annotated_image = image.copy()

# for i, contour in enumerate(contours):
#     # Calculate the area in pixels
#     pixel_area = cv2.contourArea(contour)
#     if pixel_area == 0:  # Ignore contours with zero area
#         continue

#     # Convert to real-world area in cm²
#     real_area = pixel_area / (pixels_per_cm ** 2)
#     particle_areas.append(real_area)

#     # Get the bounding box for annotation
#     x, y, w, h = cv2.boundingRect(contour)
#     annotation = f"{real_area:.2f} cm²"

#     # Draw a black rectangle behind the annotation
#     font_scale = 0.4
#     font_thickness = 1
#     text_size = cv2.getTextSize(annotation, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
#     text_width, text_height = text_size
#     cv2.rectangle(
#         annotated_image,
#         (x, y - text_height - 5),
#         (x + text_width, y),
#         (0, 0, 0),  # Black rectangle
#         -1  # Filled
#     )

#     # Draw the annotation text in white
#     cv2.putText(
#         annotated_image,
#         annotation,
#         (x, y - 5),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         font_scale,
#         (255, 255, 255),  # White text
#         font_thickness
#     )

# # Step 6: Save and Display the Annotated Image
# cv2.imwrite("annotated_shapes_with_sizes.png", annotated_image)
# cv2.imshow("Annotated Shapes", annotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Step 7: Output All Areas
# print("Particle Areas (in cm²):")
# for i, area in enumerate(particle_areas):
#     print(f"Particle {i + 1}: {area:.2f} cm²")


# import cv2
# import numpy as np

# # Load the image
# image_path = "sample_3.png"  # Replace with your actual image path
# image = cv2.imread(image_path)

# # Step 1: Convert the Image to Grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Step 2: Threshold to Create a Binary Image
# # Use the binary mask (white lines on black background)
# _, binary_mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# # Step 3: Detect Contours
# contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(f"Number of contours detected: {len(contours)}\n")

# # Step 4: Define Scale for Real-World Conversion
# scale_length_in_pixels = 500  # Replace with your scale length in pixels
# real_scale_length_cm = 100  # Known real-world length in cm
# pixels_per_cm = scale_length_in_pixels / real_scale_length_cm

# # Step 5: Calculate Areas and List All Sizes
# particle_areas = []
# for i, contour in enumerate(contours):
#     # Calculate the area in pixels
#     pixel_area = cv2.contourArea(contour)
#     if pixel_area == 0:  # Ignore contours with zero area
#         continue

#     # Convert to real-world area in cm²
#     real_area = pixel_area / (pixels_per_cm ** 2)
#     particle_areas.append(real_area)

#     # Print the area of each particle
#     print(f"Particle {i + 1}: {real_area:.2f} cm²")

# # Step 6: Annotate the Image
# annotated_image = image.copy()
# for i, contour in enumerate(contours):
#     x, y, w, h = cv2.boundingRect(contour)
#     annotation = f"{particle_areas[i]:.2f} cm²"

#     # Draw a black rectangle behind the annotation
#     font_scale = 0.4
#     font_thickness = 1
#     text_size = cv2.getTextSize(annotation, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
#     text_width, text_height = text_size
#     cv2.rectangle(
#         annotated_image,
#         (x, y - text_height - 5),
#         (x + text_width, y),
#         (0, 0, 0),  # Black rectangle
#         -1  # Filled
#     )

#     # Draw the annotation text in white
#     cv2.putText(
#         annotated_image,
#         annotation,
#         (x, y - 5),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         font_scale,
#         (255, 255, 255),  # White text
#         font_thickness
#     )

# # Step 7: Save and Display the Annotated Image
# cv2.imwrite("annotated_shapes_with_sizes_2.png", annotated_image)
# print("\nAnnotated image saved as 'annotated_shapes_with_sizes.png'")


import cv2
import numpy as np

# Load the image
image_path = "sample_3.png"  # Replace with your actual image path
image = cv2.imread(image_path)

# Step 1: Convert the Image to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Threshold to Create a Binary Image
_, binary_mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Step 3: Detect Contours
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Number of contours detected: {len(contours)}\n")

# Step 4: Define Scale for Real-World Conversion
scale_length_in_pixels = 500  # Replace with your scale length in pixels
real_scale_length_cm = 100  # Known real-world length in cm
pixels_per_cm = scale_length_in_pixels / real_scale_length_cm

# Step 5: Calculate Areas and List All Sizes
particle_areas = []
valid_contours = []  # Keep track of valid contours that correspond to areas

for contour in contours:
    # Calculate the area in pixels
    pixel_area = cv2.contourArea(contour)
    if pixel_area == 0:  # Ignore contours with zero area
        continue

    # Convert to real-world area in cm²
    real_area = pixel_area / (pixels_per_cm ** 2)
    particle_areas.append(real_area)
    valid_contours.append(contour)  # Save only valid contours

    # Print the area of each particle
    print(f"Particle {len(particle_areas)}: {real_area:.2f} cm²")

# Step 6: Annotate the Image
annotated_image = image.copy()
for i, contour in enumerate(valid_contours):  # Use valid_contours to match particle_areas
    x, y, w, h = cv2.boundingRect(contour)
    annotation = f"{particle_areas[i]:.2f} cm²"

    # Draw a black rectangle behind the annotation
    font_scale = 0.4
    font_thickness = 1
    text_size = cv2.getTextSize(annotation, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    text_width, text_height = text_size
    cv2.rectangle(
        annotated_image,
        (x, y - text_height - 5),
        (x + text_width, y),
        (0, 0, 0),  # Black rectangle
        -1  # Filled
    )

    # Draw the annotation text in white
    cv2.putText(
        annotated_image,
        annotation,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),  # White text
        font_thickness
    )

# Step 7: Save and Display the Annotated Image
cv2.imwrite("annotated_shapes_with_sizes_3.png", annotated_image)
print("\nAnnotated image saved as 'annotated_shapes_with_sizes.png'")
print(f"Number of contours detected: {len(contours)}\n")