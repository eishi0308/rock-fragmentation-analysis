import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the Segmented Image
image_path = "/mnt/data/sample1.png"  # Path to your segmented image
image = cv2.imread(image_path)

# Step 2: Detect the Blue Lines (Segmentation Boundaries)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Define HSV range for blue
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Step 3: Create a Binary Mask of Particles
# Invert the blue mask to create a particle mask
particle_mask = cv2.bitwise_not(blue_mask)

# Step 4: Detect Contours of Particles
contours, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 5: Assume a Pixel-to-Real-World Scale
# Assume 1 pixel corresponds to 0.1 cm (i.e., 10 pixels = 1 cm)
assumed_pixels_per_cm = 10  # Adjust this assumption as needed
print(f"Assumed Pixels per cm: {assumed_pixels_per_cm}")

# Step 6: Calculate the Area of Each Particle
particle_sizes_cm2 = []
output_image = image.copy()
for i, contour in enumerate(contours):
    # Calculate area in pixels
    area_pixels = cv2.contourArea(contour)
    # Convert to cm² using assumed scale
    area_cm2 = area_pixels / (assumed_pixels_per_cm ** 2)
    particle_sizes_cm2.append(area_cm2)
    # Draw contours and label particles on the image
    cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
    M = cv2.moments(contour)
    if M["m00"] != 0:  # Avoid division by zero
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(output_image, f"{i+1}: {area_cm2:.2f} cm²", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Step 7: Save and Display the Results
output_image_path = "output_particles_with_sizes.png"
cv2.imwrite(output_image_path, output_image)

print(f"Number of Particles Detected: {len(particle_sizes_cm2)}")
print("Particle Sizes in cm²:")
print(particle_sizes_cm2)

plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Particles with Real Sizes (cm²) - Assumed Scale")
plt.show()
