import numpy as np
import imageio

# Constants
OPENING_FILTER = 1
CLOSING_FILTER = 2

# Image Processing Functions

def grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale."""
    luminance_weights = np.array([0.299, 0.587, 0.114])
    return np.dot(image[..., :3], luminance_weights).astype(np.uint8)

def binarize(image: np.ndarray, threshold: float = 127.0) -> np.ndarray:
    """Binarize a grayscale image based on a threshold value."""
    return np.where(image > threshold, 1, 0)

def opening_or_closing_filter(image: np.ndarray, kernel_size: int, is_opening: bool) -> np.ndarray:
    """Apply either opening or closing morphological filter to a binary image."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    if is_opening:
        return imageio.morphology.binary_opening(image, kernel)
    else:
        return imageio.morphology.binary_closing(image, kernel)

def compute_haralick_descriptors(image: np.ndarray, q_value: tuple[int, int]) -> list[float]:
    """Compute Haralick descriptors for a binary image at a given coordinate."""
    co_occurrence_matrix = imageio.morphology.greycomatrix(image, [q_value[0]], [q_value[1]], symmetric=True, normed=True)
    return imageio.morphology.greycoprops(co_occurrence_matrix)

def euclidean_distance(descriptor1: np.ndarray, descriptor2: np.ndarray) -> float:
    """Calculate the Euclidean distance between two Haralick descriptors."""
    return np.linalg.norm(descriptor1 - descriptor2)

def rank_images_by_similarity(base_descriptor: np.ndarray, descriptors: list[np.ndarray]) -> list[tuple[int, float]]:
    """Rank images by similarity to the base image using Euclidean distances."""
    distances = [euclidean_distance(base_descriptor, descriptor) for descriptor in descriptors]
    return sorted(enumerate(distances), key=lambda x: x[1])

# Main Function

def main():
    # User input
    base_image_index = int(input("Enter the index of the base image: "))
    q_value = tuple(map(int, input("Enter the q-value (e.g., 0 1): ").split()))
    filter_type = int(input("Enter the filter type (1 for opening, 2 for closing): "))
    threshold = int(input("Enter the binarization threshold: "))
    num_images = int(input("Enter the number of images: "))

    # Read and process images
    images = [grayscale(imageio.imread(input().rstrip())) for _ in range(num_images)]
    binary_images = [binarize(image, threshold) for image in images]
    filtered_images = [opening_or_closing_filter(image, 3, filter_type == OPENING_FILTER) for image in binary_images]

    # Compute Haralick descriptors for the base image
    base_descriptor = compute_haralick_descriptors(filtered_images[base_image_index], q_value)

    # Compute Haralick descriptors for all images and rank them
    descriptors = [compute_haralick_descriptors(image, q_value) for image in filtered_images]
    ranked_images = rank_images_by_similarity(base_descriptor, descriptors)

    # Display the ranking
    print(f"Query: Image {base_image_index}")
    print("Ranking:")
    for rank, (image_index, distance) in enumerate(ranked_images):
        print(f"({rank + 1}) Image {image_index}: Distance = {distance:.4f}")

if __name__ == "__main__":
    main()
