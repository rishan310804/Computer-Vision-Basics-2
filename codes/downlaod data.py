import os
import numpy as np
from PIL import Image


def create_sample_data():
    """sample data for testing"""
    print("Creating sample cameraman image")
    os.makedirs("data", exist_ok=True)

    # Create 256x256 test image
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)

    # Add patterns (circles, squares)
    center = size // 2
    for r in range(20, 120, 15):
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= r**2
        img[mask] = 80 + r

    # Save as cameraman.jpg
    Image.fromarray(img).save("data/cameraman.jpg")
    print("Sample image created: data/cameraman.jpg")


if __name__ == "__main__":
    create_sample_data()
