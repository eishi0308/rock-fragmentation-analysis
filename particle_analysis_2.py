# import cv2
# import numpy as np

# # Load the provided image
# image_path = "image_c.png"  # Replace with the actual image path
# image = cv2.imread(image_path)

# # Convert the image to HSV for color-based segmentation
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define HSV color ranges for segmentation (these ranges are approximations and may need fine-tuning)
# color_ranges = {
#     "red": ((0, 100, 100), (10, 255, 255)),  # Red range
#     "yellow": ((20, 100, 100), (30, 255, 255)),  # Yellow range
#     "green": ((40, 50, 50), (80, 255, 255)),  # Green range
#     "blue": ((90, 50, 50), (130, 255, 255)),  # Blue range
# }

# # Scale factor: pixels per mm
# pixels_per_mm = 10  # Adjust based on the actual scale in your image

# # Dictionary to store particle sizes for each color
# particle_sizes_by_color = {}

# # Process each color range
# for color_name, (lower, upper) in color_ranges.items():
#     # Create a binary mask for the current color
#     mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))

#     # Find contours for the current mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Calculate areas for each contour and convert to mm²
#     areas = [cv2.contourArea(contour) / (pixels_per_mm ** 2) for contour in contours]

#     # Store particle sizes in the dictionary
#     particle_sizes_by_color[color_name] = areas

# # Print particle sizes for each color
# for color, sizes in particle_sizes_by_color.items():
#     print(f"Particle sizes for {color}:")
#     print(sizes)

# # Optionally, visualize segmented regions
# segmented_image = np.zeros_like(image)
# for color_name, (lower, upper) in color_ranges.items():
#     mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
#     color = (0, 0, 0)
#     if color_name == "red":
#         color = (0, 0, 255)
#     elif color_name == "yellow":
#         color = (0, 255, 255)
#     elif color_name == "green":
#         color = (0, 255, 0)
#     elif color_name == "blue":
#         color = (255, 0, 0)
#     segmented_image[mask > 0] = color

# # Display the segmented image
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
# plt.title("Segmented Regions by Color")
# plt.axis('off')
# plt.show()



import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the provided image
image_path = "image_c.png"  # Replace with the actual image path
image = cv2.imread(image_path)

# Convert the image to HSV for color-based segmentation
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define precise HSV color ranges for segmentation
color_ranges = {
    "red": ((0, 120, 70), (10, 255, 255)),  # Adjusted HSV range for red
    "yellow": ((20, 100, 100), (30, 255, 255)),  # Adjusted HSV range for yellow
    "green": ((40, 50, 50), (80, 255, 255)),  # Adjusted HSV range for green
    "blue": ((90, 50, 50), (130, 255, 255)),  # Adjusted HSV range for blue
}

# Scale factor: pixels per mm
pixels_per_mm = 10  # Adjust based on the actual scale in your image

# Dictionary to store particle sizes for each color
particle_sizes_by_color = {}

# Structuring element for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Process each color range
for color_name, (lower, upper) in color_ranges.items():
    # Create a binary mask for the current color
    mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))

    # Apply morphological operations to refine the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours for the current mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate areas for each contour and convert to mm²
    areas = [cv2.contourArea(contour) / (pixels_per_mm ** 2) for contour in contours]

    # Store particle sizes in the dictionary
    particle_sizes_by_color[color_name] = areas

    # Draw contours for visual debugging
    cv2.drawContours(image, contours, -1, (255, 255, 255), 2)

# Print particle sizes for each color
for color, sizes in particle_sizes_by_color.items():
    print(f"Particle sizes for {color}:")
    print(sizes)

# Optionally, visualize segmented regions with enhanced accuracy
segmented_image = np.zeros_like(image)
for color_name, (lower, upper) in color_ranges.items():
    mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    color = (0, 0, 0)
    if color_name == "red":
        color = (0, 0, 255)
    elif color_name == "yellow":
        color = (0, 255, 255)
    elif color_name == "green":
        color = (0, 255, 0)
    elif color_name == "blue":
        color = (255, 0, 0)
    segmented_image[mask > 0] = color

# Display the segmented image
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title("Enhanced Segmented Regions by Color")
plt.axis('off')
plt.show()
