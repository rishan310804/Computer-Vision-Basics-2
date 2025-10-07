import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def computing_integral_image_using_loop(img):
    """Compute integral image using nested loops"""
    h, w = img.shape
    I = np.zeros((h+1, w+1), dtype=np.int64)
    for r in range(1, h+1):
        row_sum = 0
        for c in range(1, w+1):
            row_sum += int(img[r-1, c-1])
            I[r, c] = I[r-1, c] + row_sum
    return I


def calculating_rectangle_sum(I, r1, c1, r2, c2):
    """Get sum of rectangular region using integral image"""
    if r1 < 0 or c1 < 0 or r2 < r1 or c2 < c1:
        raise ValueError("Invalid region coordinates")
    return int(I[r2+1, c2+1] - I[r1, c2+1] - I[r2+1, c1] + I[r1, c1])


def verify_integral(img, I, trials=100):
    """Verify integral image correctness with random testing"""
    H, W = img.shape
    failed = 0
    for _ in range(trials):
        r1 = np.random.randint(0, H)
        c1 = np.random.randint(0, W)
        r2 = np.random.randint(r1, H)
        c2 = np.random.randint(c1, W)
        s_int = calculating_rectangle_sum(I, r1, c1, r2, c2)
        s_dir = int(np.sum(img[r1:r2+1, c1:c2+1]))
        if s_int != s_dir:
            print(f" Test failed: ({r1},{c1})–({r2},{c2}): {s_int} vs {s_dir}")
            failed += 1
    print(f"✅ Verification: {trials - failed}/{trials} passed")
    return failed == 0


def compute_haar_feature(I, top, left, fw, fh, pattern):
    H, W = I.shape[0]-1, I.shape[1]-1
    if top < 0 or left < 0 or top+fh > H or left+fw > W:
        return None
    if pattern == 1:
        if fw % 2:
            return None
        m = fw//2
        a = calculating_rectangle_sum(I, top, left, top+fh-1, left+m-1)
        b = calculating_rectangle_sum(I, top, left+m, top+fh-1, left+fw-1)
        return a - b
    if pattern == 2:
        if fh % 2:
            return None
        m = fh//2
        a = calculating_rectangle_sum(I, top, left, top+m-1, left+fw-1)
        b = calculating_rectangle_sum(I, top+m, left, top+fh-1, left+fw-1)
        return a - b
    if pattern == 3:
        if fw % 3:
            return None
        q = fw//3
        a = calculating_rectangle_sum(I, top, left, top+fh-1, left+q-1)
        b = calculating_rectangle_sum(I, top, left+q, top+fh-1, left+2*q-1)
        c = calculating_rectangle_sum(I, top, left+2*q, top+fh-1, left+fw-1)
        return (a + c) - b
    raise ValueError("Unknown pattern")


def extract_haar_features(img, I, center_size=50, min_filter_size=24):
    H, W = img.shape
    r0, c0 = H//2 - center_size//2, W//2 - center_size//2
    r1, c1 = max(0, r0), max(0, c0)
    r2, c2 = min(H, r1+center_size), min(W, c1+center_size)
    features, locs = [], []
    for fs in (24, 32, 48):
        if fs > center_size:
            continue
        stride = max(1, fs//4)
        for y in range(r1, r2-fs+1, stride):
            for x in range(c1, c2-fs+1, stride):
                for p in (1, 2, 3):
                    v = compute_haar_feature(I, y, x, fs, fs, p)
                    if v is not None:
                        features.append(v)
                        locs.append((y, x, fs, p))
    return np.array(features, np.int64), locs


def draw_haar_pattern(ax, top, left, size, pattern):
    if pattern == 1:
        m = size//2
        ax.add_patch(Rectangle((left, top), m, size,
                     facecolor='green', alpha=0.3))
        ax.add_patch(Rectangle((left+m, top), m, size,
                     facecolor='red', alpha=0.3))
    elif pattern == 2:
        m = size//2
        ax.add_patch(Rectangle((left, top), size, m,
                     facecolor='green', alpha=0.3))
        ax.add_patch(Rectangle((left, top+m), size,
                     m, facecolor='red', alpha=0.3))
    elif pattern == 3:
        q = size//3
        ax.add_patch(Rectangle((left, top), q, size,
                     facecolor='green', alpha=0.3))
        ax.add_patch(Rectangle((left+q, top), q, size,
                     facecolor='orange', alpha=0.3))
        ax.add_patch(Rectangle((left+2*q, top), q, size,
                     facecolor='green', alpha=0.3))


def question1_integral_haar(image_path="data/cameraman.jpg"):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: '{image_path}' not found!")
        return
    print(f"Image loaded: {img.shape}")

    # Part (i): integral image
    I = computing_integral_image_using_loop(img)
    print(f"Integral image size: {I.shape}")
    ok = verify_integral(img, I, trials=100)

    # Part (ii): Haar features
    feats, locs = extract_haar_features(img, I)
    counts = [sum(1 for _, _, _, p in locs if p == i) for i in (1, 2, 3)]
    print(f"Total features: {len(feats)}")
    print(
        f"Pattern 1: {counts[0]}, Pattern 2: {counts[1]}, Pattern 3: {counts[2]}")

    # Visualization
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.imshow(I[1:, 1:], cmap='hot')
    plt.title("Integral")
    plt.axis('off')
    H, W = img.shape
    patch = img[H//2-25:H//2+25, W//2-25:W//2+25]
    plt.subplot(2, 3, 3)
    plt.imshow(patch, cmap='gray')
    plt.title("Center 50×50")
    plt.axis('off')
    plt.subplot(2, 3, 4)
    names = ["H2", "V2", "H3"]
    plt.bar(names, counts, color=['red', 'green', 'blue'])
    plt.title("Counts")
    plt.subplot(2, 3, 5)
    if feats.size:
        plt.hist(feats, bins=30, color='skyblue', edgecolor='black')
        plt.title("Response Dist.")
        plt.xlabel("Value")
        plt.ylabel("Freq")
    else:
        plt.text(0.5, 0.5, "No Features", ha='center', va='center')
    plt.subplot(2, 3, 6)
    ax = plt.gca()
    ax.imshow(img, cmap='gray')
    shown = [False, False, False]
    n = 0
    for y, x, s, p in locs:
        if not shown[p-1]:
            draw_haar_pattern(ax, y, x, s, p)
            shown[p-1] = True
            n += 1
        if n == 3:
            break
    plt.title("Example Filters")
    plt.axis('off')

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/question1_results.png", dpi=150)
    plt.show()

    print(f"Verification {'PASSED' if ok else 'FAILED'}")
    print(f"Results saved to results/question1_results.png")


if __name__ == "__main__":
    question1_integral_haar()
