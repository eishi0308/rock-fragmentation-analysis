import cv2
import numpy as np

# Load the image
image_path = "sample_3.png"  # Replace with the actual image path
image = cv2.imread(image_path)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


##step2
# Convert to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the HSV range for blue
lower_blue = np.array([100, 150, 50])  # Adjust if needed
upper_blue = np.array([140, 255, 255])

# Create a binary mask for blue lines
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)



#step3
blue_lines = cv2.bitwise_and(image, image, mask=blue_mask)
cv2.imshow("Blue Lines", blue_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()


# #step4
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# refined_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)


# #step5
# edges = cv2.Canny(refined_mask, 50, 150)



# #Step 4: Detect Contours
# contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# contour_image = image.copy()
# cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# # Display the image with contours
# cv2.imshow("Contours", contour_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# # Step 5: Calculate Particle Areas

# scale_length_in_pixels = 500  # Measure this using an image viewer
# real_scale_length_cm = 100  # The scale indicates 100 cm
# pixels_per_cm = scale_length_in_pixels / real_scale_length_cm


# particle_areas = []
# for contour in contours:
#     pixel_area = cv2.contourArea(contour)
#     real_area = pixel_area / (pixels_per_cm ** 2)  # Convert to cm²
#     particle_areas.append(real_area)

# # Print the particle areas
# for i, area in enumerate(particle_areas):
#     print(f"Particle {i + 1}: {area:.2f} cm²")



# for i, contour in enumerate(contours):
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.putText(contour_image, f"{particle_areas[i]:.2f} cm²", (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# # Save the annotated image
# cv2.imwrite("annotated_particles.png", contour_image)


# import pandas as pd

# df = pd.DataFrame({"Particle ID": range(1, len(particle_areas) + 1), "Area (cm²)": particle_areas})
# df.to_csv("particle_sizes.csv", index=False)



############################################################################################################


# #

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# refined_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

# # Display the refined mask for debugging
# cv2.imshow("Refined Mask", refined_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #

# contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Create a copy of the original image for annotation
# contour_image = image.copy()

# # Draw contours for visualization
# cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Green contours
# cv2.imshow("Contours", contour_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# print(f"Number of contours detected: {len(contours)}")


# cv2.imshow("Blue Mask", blue_mask)
# cv2.imshow("Refined Mask", refined_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()







# import cv2
# import numpy as np

# # Load the image
# image_path = "sample_2.png"  # Replace with the actual image path
# image = cv2.imread(image_path)

# # Display original image
# cv2.imshow("Original Image", image)
# cv2.waitKey(0)

# # Step 1: Convert to HSV to Isolate Blue Lines
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define the HSV range for blue
# lower_blue = np.array([100, 150, 50])  # Adjust if needed
# upper_blue = np.array([140, 255, 255])

# # Create a binary mask for blue lines
# blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# # Display the blue mask for validation
# cv2.imshow("Blue Mask", blue_mask)
# cv2.waitKey(0)

# # Step 2: Refine the Mask Using Morphological Operations
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# refined_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

# # Display the refined mask for validation
# cv2.imshow("Refined Mask", refined_mask)
# cv2.waitKey(0)

# # Step 3: Detect Edges (Optional for Debugging)
# edges = cv2.Canny(refined_mask, 50, 150)

# # Display edges for validation
# cv2.imshow("Edges Detected", edges)
# cv2.waitKey(0)

# # Step 4: Detect Contours of Particles
# contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Create a copy of the image to visualize contours
# contour_image = image.copy()
# cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# # Display contours for validation
# cv2.imshow("Contours", contour_image)
# cv2.waitKey(0)

# # Step 5: Define the Scale for Real-World Conversion
# scale_length_in_pixels = 500  # Replace with the actual pixel length of the scale in your image
# real_scale_length_cm = 100  # The known real-world length of the scale (e.g., 100 cm)
# pixels_per_cm = scale_length_in_pixels / real_scale_length_cm

# # Step 6: Calculate Particle Areas and Annotate
# particle_areas = []

# for i, contour in enumerate(contours):
#     # Calculate the area in pixels
#     pixel_area = cv2.contourArea(contour)
    
#     # Convert the area to cm²
#     real_area = pixel_area / (pixels_per_cm ** 2)
#     particle_areas.append(real_area)

#     # Get the bounding box for placing text
#     x, y, w, h = cv2.boundingRect(contour)
    
#     # Create annotation text
#     annotation = f"{real_area:.2f} cm²"
    
#     # Draw a black background rectangle for text
#     font_scale = 0.5
#     font_thickness = 1
#     text_size = cv2.getTextSize(annotation, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
#     text_width, text_height = text_size
#     cv2.rectangle(
#         contour_image,
#         (x, y - text_height - 5),
#         (x + text_width, y),
#         (0, 0, 0),  # Black rectangle
#         -1  # Fill the rectangle
#     )
    
#     # Draw the annotation text in white
#     cv2.putText(
#         contour_image,
#         annotation,
#         (x, y - 5),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         font_scale,
#         (255, 255, 255),  # White text color
#         font_thickness
#     )

# # Save and Display the Final Annotated Image
# cv2.imwrite("annotated_particles_blue_only.png", contour_image)
# cv2.imshow("Annotated Image", contour_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Step 7: Output Particle Sizes to Console
# print("Particle Sizes (in cm²):")
# for i, area in enumerate(particle_areas):
#     print(f"Particle {i + 1}: {area:.2f} cm²")



















# for i, contour in enumerate(contours):
#     # Get the bounding box of the contour
#     x, y, w, h = cv2.boundingRect(contour)

#     # Prepare the text for annotation
#     text = f"{particle_areas[i]:.2f} cm²"
#     font_scale = 0.5
#     font_thickness = 1

#     # Calculate text size for background rectangle
#     text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
#     text_width, text_height = text_size

#     # Draw a black rectangle for the background
#     cv2.rectangle(
#         contour_image,
#         (x, y - text_height - 5),  # Top-left corner of the rectangle
#         (x + text_width, y),  # Bottom-right corner of the rectangle
#         (0, 0, 0),  # Black background
#         -1  # Fill the rectangle
#     )

#     # Draw the text in white on top of the rectangle
#     cv2.putText(
#         contour_image,
#         text,
#         (x, y - 5),  # Position the text slightly above the bounding box
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

# print(particle_areas)
# print(len(contours))
