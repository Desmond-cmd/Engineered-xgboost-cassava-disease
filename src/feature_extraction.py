import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os

def extract_hsv_histogram(image, bins=36):
    """
    Extract HSV colour histogram from cassava leaf image.
    Captures colour distribution which differs between healthy
    and diseased leaves.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    return np.concatenate([h_hist, s_hist, v_hist])


def extract_glcm_features(image):
    """
    Extract GLCM texture features from cassava leaf image.
    Texture patterns differ between disease types.
    Returns: contrast, correlation, energy, homogeneity
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(
        gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
        256, symmetric=True, normed=True
    )
    features = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        values = graycoprops(glcm, prop).flatten()
        features.extend(values)
    return np.array(features)


def extract_canny_features(image):
    """
    Extract edge density features using Canny edge detection.
    Disease lesions create distinct edge patterns on leaves.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    edge_mean = np.mean(edges)
    edge_std = np.std(edges)
    return np.array([edge_density, edge_mean, edge_std])


def extract_all_features(image_path):
    """
    Extract all features from a single cassava leaf image.
    Combines HSV histogram + GLCM texture + Canny edges.

    Parameters
    ----------
    image_path : str
        Path to the image file (.jpg)

    Returns
    -------
    np.array
        Feature vector of shape (123,)
        - 108 HSV histogram features (36 bins x 3 channels)
        - 16 GLCM texture features (4 properties x 4 angles)
        - 3 Canny edge features
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (224, 224))
    hsv_features = extract_hsv_histogram(img)
    glcm_features = extract_glcm_features(img)
    canny_features = extract_canny_features(img)
    all_features = np.concatenate([
        hsv_features,
        glcm_features,
        canny_features
    ])
    return all_features


def build_feature_matrix(image_dir, labels_df, image_col='image_id',
                          label_col='label'):
    """
    Build feature matrix from all cassava leaf images.

    Parameters
    ----------
    image_dir : str
        Path to folder containing cassava images
    labels_df : pd.DataFrame
        DataFrame with image filenames and labels
    image_col : str
        Column name for image filenames
    label_col : str
        Column name for disease labels

    Returns
    -------
    X : np.array
        Feature matrix of shape (n_samples, n_features)
    y : np.array
        Label array of shape (n_samples,)
    failed : list
        List of images that could not be processed
    """
    X = []
    y = []
    failed = []

    total = len(labels_df)
    for i, row in labels_df.iterrows():
        image_path = os.path.join(image_dir, row[image_col])
        try:
            features = extract_all_features(image_path)
            X.append(features)
            y.append(row[label_col])
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{total} images")
        except Exception as e:
            failed.append((row[image_col], str(e)))

    print(f"Done. Processed: {total - len(failed)}, Failed: {len(failed)}")
    return np.array(X), np.array(y), failed
