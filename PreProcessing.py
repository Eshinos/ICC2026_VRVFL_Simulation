import pickle
import torch
import numpy as np
import os

DATA_DIR = "./cifar-10-batches-py"
SAVE_DIR = "./cifar10_tensors"

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Function to load a batch
def load_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch[b'data'], batch[b'labels']

# Load training data
train_images, train_labels = [], []

for i in range(1, 6):  # There are 5 training batches
    batch_file = os.path.join(DATA_DIR, f"data_batch_{i}")
    data, labels = load_batch(batch_file)
    train_images.append(data)
    train_labels.extend(labels)

train_images = np.vstack(train_images).reshape(-1, 3, 32, 32)  # Reshape to (N, C, H, W)
train_labels = np.array(train_labels)

# Load test data
test_file = os.path.join(DATA_DIR, "test_batch")
test_images, test_labels = load_batch(test_file)
test_images = test_images.reshape(-1, 3, 32, 32)
test_labels = np.array(test_labels)

# Convert to PyTorch tensors
train_images_tensor = torch.tensor(train_images, dtype=torch.float32) / 255.0  # Normalize to [0,1]
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_images_tensor = torch.tensor(test_images, dtype=torch.float32) / 255.0
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# Save tensors
torch.save(train_images_tensor, os.path.join(SAVE_DIR, "train_images.pt"))
torch.save(train_labels_tensor, os.path.join(SAVE_DIR, "train_labels.pt"))
torch.save(test_images_tensor, os.path.join(SAVE_DIR, "test_images.pt"))
torch.save(test_labels_tensor, os.path.join(SAVE_DIR, "test_labels.pt"))

print("CIFAR-10 dataset processed and saved as tensors!")