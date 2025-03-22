import cv2
import numpy as np
import os

def merge_and_blur(image_path1, image_path2, output_path):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    if img1 is None or img2 is None:
        print("Error: One or both images could not be loaded.")
        return
    
    height = min(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (img1.shape[1], height))
    img2 = cv2.resize(img2, (img2.shape[1], height))
    
    blend_width = min(50, img1.shape[1], img2.shape[1])
    total_width = img1.shape[1] + img2.shape[1] - blend_width
    merged_image = np.zeros((height, total_width, 3), dtype=np.uint8)
    
    merged_image[:, :img1.shape[1] - blend_width] = img1[:, :img1.shape[1] - blend_width]
    merged_image[:, img1.shape[1]:] = img2[:, blend_width:]
    
    for i in range(blend_width):
        alpha = i / blend_width
        merged_image[:, img1.shape[1] - blend_width + i] = (
            (1 - alpha) * img1[:, img1.shape[1] - blend_width + i] + alpha * img2[:, i]
        ).astype(np.uint8)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, merged_image)
    print(f"Merged image saved to {output_path}")

# Example usage:
image1 = "../data/images/white-american-woman_programming.png"
image2 = "../data/images/white-american-woman_babysitting.png"
output_image = "../data/images/woman_babysitting.png"
merge_and_blur(image1, image2, output_image)