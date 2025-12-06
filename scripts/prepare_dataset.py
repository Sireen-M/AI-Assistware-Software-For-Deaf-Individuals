import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from collections import Counter
from tqdm import tqdm

# Paths
LANDMARKS_DIR = 'landmarks_csv'
OUTPUT_DIR = 'scripts'
TOP_N_CLASSES = 20  # Change this to limit more or less

X, y = [], []
labels = []

# Collect all CSVs and labels
for file in os.listdir(LANDMARKS_DIR):
    if not file.endswith('.csv'):
        continue
    label = file.split('_')[0]  # Assuming file name format: SIGN_LABEL_123.csv
    labels.append(label)

# Count frequencies and get top N
label_counts = Counter(labels)
top_labels = set([label for label, _ in label_counts.most_common(TOP_N_CLASSES)])

print(f"🎯 Keeping top {TOP_N_CLASSES} classes: {sorted(top_labels)}")

# Reload only selected files
for file in tqdm(os.listdir(LANDMARKS_DIR)):
    if not file.endswith('.csv'):
        continue
    label = file.split('_')[0]
    if label not in top_labels:
        continue

    filepath = os.path.join(LANDMARKS_DIR, file)
    data = pd.read_csv(filepath).values

    # Pad or trim to 30 frames
    if len(data) < 30:
        pad = np.zeros((30 - len(data), data.shape[1]))
        data = np.vstack((data, pad))
    elif len(data) > 30:
        data = data[:30]

    X.append(data)
    y.append(label)

# Convert to arrays
X = np.array(X)
y = np.array(y)

# One-hot encode labels
encoder = LabelBinarizer()
y_encoded = encoder.fit_transform(y)
classes = encoder.classes_

# Save
np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y_encoded)
np.save(os.path.join(OUTPUT_DIR, 'classes.npy'), classes)

print(f"\n✅ Dataset prepared successfully with top {TOP_N_CLASSES} classes:")
print(f"   ➤ X shape: {X.shape}")
print(f"   ➤ y shape: {y_encoded.shape}")
print(f"   ➤ Classes: {len(classes)} | {list(classes)}")