import os
import glob
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Configuration
IMG_SIZE = 128
BOW_N_CLUSTERS = 100
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def question2_texture_classification():
    print("QUESTION 2: KTH-TIPS Texture Classification ")

    # Load KTH-TIPS dataset
    def load_kth_tips(data_path="data/KTH-TIPS"):
        if not os.path.exists(data_path):
            print(f"KTH-TIPS not found at {data_path}")
            return None, None, None

        images, labels, class_names = [], [], []
        class_dirs = sorted([d for d in os.listdir(data_path)
                             if os.path.isdir(os.path.join(data_path, d))])

        for class_id, class_name in enumerate(class_dirs):
            class_path = os.path.join(data_path, class_name)
            class_names.append(class_name)

            image_files = []
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
                image_files.extend(glob.glob(os.path.join(class_path, ext)))

            for img_path in image_files:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    images.append(img_resized)
                    labels.append(class_id)

        return np.array(images, dtype=np.uint8), np.array(labels), class_names

    # PART (i): Raw Pixel Features
    def extract_raw_features(images):
        start_time = time.time()
        features = images.reshape(len(images), -1).astype(np.float32) / 255.0
        return features, time.time() - start_time

    # PART (ii): LBP Features
    def computing_lbp(image):
        h, w = image.shape
        lbp = np.zeros((h, w), dtype=np.uint8)

        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                code = 0
                neighbors = [image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                             image[i, j+1], image[i+1, j+1], image[i+1, j],
                             image[i+1, j-1], image[i, j-1]]

                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)

                lbp[i, j] = code
        return lbp

    def extract_lbp_features(images):
        start_time = time.time()
        features = []
        for img in images:
            lbp_img = computing_lbp(img)
            hist, _ = np.histogram(lbp_img.flatten(), bins=256, range=(0, 256))
            hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
            features.append(hist)
        return np.array(features), time.time() - start_time

    # PART (iii): BoW Features
    def extract_bow_features(images):
        start_time = time.time()
        all_descriptors = []
        image_descriptors = []

        for img in images:
            corners = cv2.goodFeaturesToTrack(img, maxCorners=50,
                                              qualityLevel=0.01, minDistance=8)
            descriptors = []

            if corners is not None:
                for corner in corners:
                    x, y = int(corner[0][0]), int(corner[0][1])
                    if 8 <= x < IMG_SIZE-8 and 8 <= y < IMG_SIZE-8:
                        patch = img[y-8:y+8, x-8:x+8].flatten()
                        descriptors.append(patch.astype(np.float32))

            if descriptors:
                descriptors = np.array(descriptors)
                image_descriptors.append(descriptors)
                all_descriptors.extend(descriptors)
            else:
                image_descriptors.append(np.array([]))

        if not all_descriptors:
            return np.zeros((len(images), BOW_N_CLUSTERS)), time.time() - start_time

        # KMeans clustering
        all_descriptors = np.array(all_descriptors)
        kmeans = KMeans(n_clusters=BOW_N_CLUSTERS, random_state=42, n_init=10)
        kmeans.fit(all_descriptors)

        bow_features = []
        for descriptors in image_descriptors:
            if len(descriptors) > 0:
                words = kmeans.predict(descriptors)
                hist = np.bincount(words, minlength=BOW_N_CLUSTERS)
                hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
            else:
                hist = np.zeros(BOW_N_CLUSTERS, dtype=np.float32)
            bow_features.append(hist)

        return np.array(bow_features), time.time() - start_time

    # PART (iv): HoG Features
    def compute_hog(image):
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * 180 / np.pi
        orientation[orientation < 0] += 180

        cell_size = 8
        n_bins = 9
        cells_y, cells_x = IMG_SIZE // cell_size, IMG_SIZE // cell_size

        hog_vector = []
        cell_hist = np.zeros((cells_y, cells_x, n_bins), dtype=np.float32)

        # Create cell histograms
        for cy in range(cells_y):
            for cx in range(cells_x):
                y0, y1 = cy*cell_size, (cy+1)*cell_size
                x0, x1 = cx*cell_size, (cx+1)*cell_size

                cell_mag = magnitude[y0:y1, x0:x1]
                cell_ori = orientation[y0:y1, x0:x1]

                hist = np.zeros(n_bins, dtype=np.float32)
                for yy in range(cell_size):
                    for xx in range(cell_size):
                        bin_idx = int(cell_ori[yy, xx] * n_bins / 180) % n_bins
                        hist[bin_idx] += cell_mag[yy, xx]

                cell_hist[cy, cx, :] = hist

        # Block normalization
        for cy in range(cells_y - 1):
            for cx in range(cells_x - 1):
                block = cell_hist[cy:cy+2, cx:cx+2, :].flatten()
                norm = np.linalg.norm(block)
                if norm > 0:
                    block = block / norm
                hog_vector.extend(block)

        return np.array(hog_vector, dtype=np.float32), magnitude, orientation, cell_hist

    def extract_hog_features(images):
        start_time = time.time()
        features = []
        sample_mag, sample_ori, sample_img, sample_cell_hist = None, None, None, None

        for i, img in enumerate(images):
            hog_feat, mag, ori, cell_hist = compute_hog(img)
            features.append(hog_feat)

            if i == 0:  # for visualization
                sample_mag, sample_ori, sample_img, sample_cell_hist = mag, ori, img, cell_hist

        return np.array(features), time.time() - start_time, sample_mag, sample_ori, sample_img, sample_cell_hist

    # Classification function
    def train_classifier(X, y, method_name):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        start_time = time.time()
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time

        test_pred = svm.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        print(f"{method_name}: {test_acc:.3f} (train: {training_time:.2f}s)")

        return test_acc, training_time

    # Main execution
    images, labels, class_names = load_kth_tips()

    if images is None:
        print("Please extract KTH-TIPS to data/KTH-TIPS/")
        return None

    print(f" Dataset: {len(images)} images, {len(class_names)} classes")

    # Sample images visualization
    plt.figure(figsize=(12, 3))
    for i in range(min(len(class_names), 10)):
        plt.subplot(1, 10, i+1)
        class_indices = np.where(labels == i)[0]
        if len(class_indices) > 0:
            plt.imshow(images[class_indices[0]], cmap='gray')
            plt.title(class_names[i][:8], fontsize=8)
            plt.axis('off')
    plt.suptitle('KTH-TIPS Sample Images')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/samples.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Extract features
    print("\nExtracting features")
    raw_features, raw_time = extract_raw_features(images)
    lbp_features, lbp_time = extract_lbp_features(images)
    bow_features, bow_time = extract_bow_features(images)
    hog_features, hog_time, sample_mag, sample_ori, sample_img, sample_cell_hist = extract_hog_features(
        images)

    # HoG visualization
    plt.figure(figsize=(15, 3))

    plt.subplot(1, 5, 1)
    plt.imshow(sample_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(sample_mag, cmap='hot')
    plt.title('Gradient Magnitude')
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(sample_ori, cmap='hsv')
    plt.title('Gradient Orientations')
    plt.axis('off')

    plt.subplot(1, 5, 4)
    feature_map = np.sum(sample_cell_hist, axis=2)
    plt.imshow(feature_map, cmap='viridis')
    plt.title('HoG Feature Map')
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.plot(hog_features[0][:100])
    plt.title('HoG Feature Vector')
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)

    plt.suptitle('HoG: Gradients & Feature Map Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/hog_gradients.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    # Train classifiers
    print("\nTraining classifiers")
    results = {}
    results['raw'] = train_classifier(raw_features, labels, "Raw Pixels")
    results['lbp'] = train_classifier(lbp_features, labels, "LBP")
    results['bow'] = train_classifier(bow_features, labels, "BoW")
    results['hog'] = train_classifier(hog_features, labels, "HoG")

    # results visualization
    methods = ['Raw', 'LBP', 'BoW', 'HoG']
    accuracies = [results['raw'][0], results['lbp']
                  [0], results['bow'][0], results['hog'][0]]
    times = [raw_time, lbp_time, bow_time, hog_time]
    dims = [raw_features.shape[1], lbp_features.shape[1],
            bow_features.shape[1], hog_features.shape[1]]
    colors = ['red', 'green', 'blue', 'orange']

    plt.figure(figsize=(15, 10))

    # Performance comparison
    plt.subplot(2, 3, 1)
    bars = plt.bar(methods, accuracies, color=colors)
    plt.ylabel('Accuracy')
    plt.title('Classification Performance')
    plt.ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    # Extraction time
    plt.subplot(2, 3, 2)
    plt.bar(methods, times, color=colors)
    plt.ylabel('Time (s)')
    plt.title('Feature Extraction Time')

    # Feature dimensions
    plt.subplot(2, 3, 3)
    plt.bar(methods, dims, color=colors)
    plt.ylabel('Dimensions')
    plt.title('Feature Vector Sizes')
    plt.yscale('log')

    # LBP example
    plt.subplot(2, 3, 4)
    lbp_example = computing_lbp(images[0])
    plt.imshow(lbp_example, cmap='gray')
    plt.title('LBP Example')
    plt.axis('off')

    # Raw pixel limitations analysis
    plt.subplot(2, 3, 5)
    plt.axis('off')
    limitations = f"""RAW PIXEL LIMITATIONS:
• High dimensional: {dims[0]:,} features
• Illumination sensitive
• No spatial invariance  
• Global representation only
• Accuracy: {accuracies[0]:.3f}"""
    plt.text(0.1, 0.8, limitations, fontsize=10, va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    plt.title('Part (i): Raw Pixels Analysis')

    # Summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    best_method = methods[np.argmax(accuracies)]
    best_acc = max(accuracies)
    summary = f"""RESULTS SUMMARY:
• Best: {best_method} ({best_acc:.3f})
• Dataset: {len(images)} images, {len(class_names)} classes
• Training: 70% split

All Accuracies:
Raw: {accuracies[0]:.3f} | LBP: {accuracies[1]:.3f}
BoW: {accuracies[2]:.3f} | HoG: {accuracies[3]:.3f}"""
    plt.text(0.1, 0.8, summary, fontsize=10, va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.title('Complete Analysis')

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nQUESTION 2 COMPLETED")
    print(f"Best method: {best_method} ({best_acc:.3f})")
    print(f"Results saved to {RESULTS_DIR}/ folder")

    return results


if __name__ == "__main__":
    question2_texture_classification()
