from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn import cluster
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

def load_dataset(dataset_dir_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i, class_dir in enumerate(sorted(dataset_dir_path.iterdir())):
        for file in class_dir.iterdir():
            img_file = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            x.append(img_file)
            y.append(i)

    return np.asarray(x), np.asarray(y)


def convert_descriptor_to_histogram(descriptors, vocab_model, normalize=True) -> np.ndarray:
    features_words = vocab_model.predict(descriptors)
    histogram = np.zeros(vocab_model.n_clusters, dtype=np.float32)
    unique, counts = np.unique(features_words, return_counts=True)
    histogram[unique] += counts
    if normalize:
        histogram /= histogram.sum()
    return histogram


def apply_feature_transform(
        data: np.ndarray,
        feature_detector_descriptor,
        vocab_model
) -> np.ndarray:
    data_transformed = []
    for image in data:
        keypoints, image_descriptors = feature_detector_descriptor.detectAndCompute(image, None)
        bow_features_histogram = convert_descriptor_to_histogram(image_descriptors, vocab_model)
        data_transformed.append(bow_features_histogram)
    return np.asarray(data_transformed)


if __name__ == '__main__':
    np.random.seed(42)

    path = Path("train")
    images, labels = load_dataset(path)

    feature_detector_descriptor = cv2.ORB_create(nfeatures=2000)

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.9,
                                                                            random_state=42)
    train_descriptors = []
    for image in train_images:
        for descriptor in feature_detector_descriptor.detectAndCompute(image, None)[1]:
            train_descriptors.append(descriptor)

    NB_WORDS = 500

    kmeans = cluster.KMeans(n_clusters=NB_WORDS, random_state=42)
    kmeans.fit(train_descriptors)

    X_train = apply_feature_transform(train_images, feature_detector_descriptor, kmeans)
    y_train = train_labels

    X_test = apply_feature_transform(test_images, feature_detector_descriptor, kmeans)
    y_test = test_labels

    # param_grid = {
    #     'verbose': [2, 5, 10, 15, 20, 25, 30],
    #     'kernel': [‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’]
    # }

    # k_fold = StratifiedKFold(n_splits=5)

    # grid_search = GridSearchCV(classifier, param_grid, cv=k_fold)
    # grid_search.fit(X_train, y_train)

    classifier = SVC(verbose=20, random_state=42, kernel='poly')
    classifier.fit(X_train, y_train)

    print(classifier.score(X_test, y_test))

    pickle.dump(classifier, open('./clf.p', 'wb'))
    pickle.dump(kmeans, open('vocab_model.p.p', 'wb'))
