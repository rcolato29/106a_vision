import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

IMAGE_NAME = sys.argv[1]
OUTPUT_DIM = int(sys.argv[2])

# load in the image as grayscale
im_gray = cv2.imread(f'images/{IMAGE_NAME}.png', cv2.IMREAD_GRAYSCALE)

# crop the image to just the whiteboard part (fixed size)
height, width = im_gray.shape
x0, y0 = width // 5, height // 5
x1, y1 = width - width // 5, height - height // 5
im_cropped = im_gray[y0 : y1, x0 : x1]

# create binary mask on blurred image (removes noise)
im_blurred = cv2.GaussianBlur(im_cropped, ksize=(5, 5), sigmaX=0)
_, im_binary = cv2.threshold(im_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# crop to smallest rectangle that encompasses drawn object
ys, xs = np.where(im_binary == 0)
min_x, min_y = min(xs), min(ys)
max_x, max_y = max(xs), max(ys)
im_rect = im_binary[min_y : max_y + 1, min_x : max_x + 1]

# pool rectangle to square of size OUTPUT_DIM x OUTPUT_DIM
im_norms = im_rect.astype(np.float32) / 255.0
im_resized = cv2.resize(im_norms, dsize=(OUTPUT_DIM, OUTPUT_DIM), interpolation=cv2.INTER_AREA)
im_matrix = (im_resized < 0.8).astype(int)
print(im_matrix)

# display the resulting image and save
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(1 - im_matrix, cmap='gray')
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f'images/{IMAGE_NAME}{OUTPUT_DIM}-output.png', 
            bbox_inches='tight', pad_inches=0, dpi=150)
plt.show()