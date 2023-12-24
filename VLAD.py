import cv2
import os
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard

# Function to extract local SIFT features from an image
def extract_sift_features(image_path):
    """
    Extracts local SIFT features from an image.

    Args:
        image_path (str): Path to the image.

    Returns:
        tuple: Tuple containing key points and descriptors.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(image, None)
    return key_points, descriptors

# Function to get all image paths from subdirectories of the dataset directory
def get_all_image_paths(directory):
    """
    Retrieves all image paths from subdirectories of the given directory.

    Args:
        directory (str): Path to the dataset directory.

    Returns:
        list: List of image paths.
    """
    image_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_paths.append(os.path.join(root, file))

    return image_paths

# Path to the directory containing the Paris dataset
image_directory = "/content/paris"

# Get all image paths from subdirectories
image_paths = get_all_image_paths(image_directory)

# Check and remove non-image files
image_paths = [path for path in image_paths if path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

# Split the dataset into training and testing sets
train_ratio = 0.8
num_train = int(train_ratio * len(image_paths))
random.shuffle(image_paths)
train_paths = image_paths[:num_train]
test_paths = image_paths[num_train:]

# Number of visual words in the codebook (k)
k = 64

# Extract local features from all images
all_features = []
for path in image_paths:
    key_points, descriptors = extract_sift_features(path)
    if descriptors is not None:
        all_features.extend(descriptors)

# Convert the list of features to a NumPy array
all_features = np.array(all_features)

# Use K-Means to create a codebook
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(all_features)

# The image dictionary (codebook) is the cluster centers
codebook = kmeans.cluster_centers_

# Function to compute VLAD vector for an image
def compute_vlad_vector(image_path, codebook):
    """
    Computes the VLAD vector for an image.

    Args:
        image_path (str): Path to the image.
        codebook (numpy.ndarray): Codebook or cluster centers.

    Returns:
        numpy.ndarray: VLAD vector.
    """
    key_points, descriptors = extract_sift_features(image_path)
    vlad_vector = np.zeros((k, descriptors.shape[1]), dtype=np.float32)

    # Assign local features to the nearest visual words
    labels = kmeans.predict(descriptors)

    for i in range(len(key_points)):
        vlad_vector[labels[i]] += descriptors[i] - codebook[labels[i]]

    # Normalize the VLAD vector using L2-norm (Euclidean norm)
    vlad_vector = vlad_vector.flatten()
    vlad_vector /= np.linalg.norm(vlad_vector)

    return vlad_vector

# Compute VLAD vectors for all images in the list
vlad_vectors = []
for path in image_paths:
    vlad_vector = compute_vlad_vector(path, codebook)
    vlad_vectors.append(vlad_vector)

# Convert the list of VLAD vectors to a NumPy array
vlad_vectors = np.array(vlad_vectors)

# Number of dimensions after reducing data dimensions using PCA
compo = len(image_paths)
n_components = compo

# Apply PCA to reduce data dimensions for both training and testing sets
pca = PCA(n_components=n_components)
all_vlad_vectors_reduced = pca.fit_transform(vlad_vectors)

# Split back into training and testing sets
train_vlad_vectors_reduced = all_vlad_vectors_reduced[:num_train]
test_vlad_vectors_reduced = all_vlad_vectors_reduced[num_train:]

# Build a KD-Tree for the training set
train_kdtree = cKDTree(train_vlad_vectors_reduced)

# Function to display image with its descriptor vector
def display_image_with_vector(image_path, vlad_vector, title):
    """
    Displays an image with its descriptor vector.

    Args:
        image_path (str): Path to the image.
        vlad_vector (numpy.ndarray): VLAD vector.
        title (str): Title for the plot.
    """
    image = cv2.imread(image_path)
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

# Function to find and display similar images
def search_similar_images(query_vector, kdtree, vlad_vectors_reduced,
                          num_results=1):
    """
    Finds and displays similar images based on a query vector.

    Args:
        query_vector (numpy.ndarray): Query vector.
        kdtree: KD-Tree built on the training set.
        vlad_vectors_reduced: Reduced VLAD vectors for the training set.
        num_results (int): Number of similar images to display.
    """
    _, indices = kdtree.query(query_vector, num_results)

    if num_results == 1:
        idx = int(indices[0])  # Access the first index
        image_path = image_paths[idx]
        display_image_with_vector(image_path,
                        vlad_vectors_reduced[idx], "Similar Image")
    else:
        for i in range(num_results):
            idx = int(indices[i][0])
            image_path = image_paths[idx]
            display_image_with_vector(image_path,
                vlad_vectors_reduced[idx], f"Similar Image {i + 1}")

# Function to find the path of a similar image
def search_similar_images_path(query_vector, kdtree, vlad_vectors_reduced,
                          num_results=1):
    """
    Finds the path of a similar image based on a query vector.

    Args:
        query_vector (numpy.ndarray): Query vector.
        kdtree: KD-Tree built on the training set.
        vlad_vectors_reduced: Reduced VLAD vectors for the training set.
        num_results (int): Number of similar images to find.

    Returns:
        str: Path of the similar image.
    """
    _, indices = kdtree.query(query_vector, num_results)

    if num_results == 1:
        idx = int(indices[0])  # Access the first index
        image_path = image_paths[idx]
        return image_path
    else:
        for i in range(num_results):
            idx = int(indices[i][0])
            image_path = image_paths[idx]
            return image_path

# Create a descriptor vector for a randomly chosen image in the test set
query_image_path = random.choice(test_paths)
