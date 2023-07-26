import os
import numpy as np
import simulation


# Generate some random images
input_images, target_masks = simulation.generate_random_data(192, 192, count=1200)

# Create files for train and test images
outDir1 = 'train_input/'
os.makedirs(outDir1, exist_ok=True)
outDir2 = 'train_target/'
os.makedirs(outDir2, exist_ok=True)
outDir3 = 'test_input/'
os.makedirs(outDir3, exist_ok=True)
outDir4 = 'test_target/'
os.makedirs(outDir4, exist_ok=True)

# Put 1000 images into train file and 200 images into test file
for i in range(1000):
    input_images_name = outDir1 + str(i) + ".npy"
    np.save(input_images_name, input_images[i])
    taget_masks_name = outDir2 + str(i) + ".npy"
    np.save(taget_masks_name, target_masks[i])

for i in range(1000,1200):
    input_images_name = outDir3 + str(i-1000) + ".npy"
    np.save(input_images_name, input_images[i])
    taget_masks_name = outDir4 + str(i-1000) + ".npy"
    np.save(taget_masks_name, target_masks[i])

