import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "blue_mask_2.png"  # Replace with the actual image path
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

# Step 2: Find contours in the mask image
contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank canvas to draw contours
contour_canvas = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)  # Convert to color for visualization

# Step 3: Display contours for selection
for i, contour in enumerate(contours):
    # Draw each contour with its index
    cv2.drawContours(contour_canvas, [contour], -1, (0, 255, 0), 2)
    # Put index label near the contour
    x, y = contour[0][0]
    cv2.putText(contour_canvas, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Save or display the image with contours for reference
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(contour_canvas, cv2.COLOR_BGR2RGB))
plt.title("Contours with Indices")
plt.axis("off")
plt.show()

# Step 4: Interactive Selection
# Ask the user to select a contour by index
selected_index = int(input("Enter the index of the shape to highlight: "))

# Validate index
if 0 <= selected_index < len(contours):
    # Highlight the selected contour
    highlighted_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(highlighted_image, [contours[selected_index]], -1, (0, 0, 255), 3)  # Highlight in red

    # Display the highlighted shape
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Highlighted Shape - Index {selected_index}")
    plt.axis("off")
    plt.show()
else:
    print("Invalid index. Please try again.")
