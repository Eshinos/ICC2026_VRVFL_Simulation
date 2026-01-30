import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
import copy
from constants import NOF_SAMPLES_PER_CLASS
import torch.nn.functional as F
import numpy as np

def get_dataloader_from_tensors(data_tensor, labels_tensor, indices, batch_size):
    """
    Creates a DataLoader from saved tensors.
    - data_tensor: Full dataset tensor (N, C, H, W)
    - labels_tensor: Full labels tensor (N,)
    - indices: List of indices for this client
    - batch_size: Batch size for training
    """
    subset_data = TensorDataset(data_tensor[indices], labels_tensor[indices])
    return DataLoader(subset_data, batch_size=batch_size, shuffle=False)

class CNN_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_CIFAR10, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )

        # Residual Block 1
        self.res1 = self._make_residual_block(32)

        # Downsampling 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        self.skip1 = nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False)

        # Residual Block 2
        self.res2 = self._make_residual_block(64)

        # Downsampling 2
        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False)
        )
        self.skip2 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        x = self.conv1(x)

        # Residual Block 1
        identity = x
        out = self.res1(x)
        x = F.relu(out + identity, inplace=False)

        # Downsampling 1
        identity = self.skip1(x)
        x = self.trans1(x) + identity
        x = F.relu(x, inplace=False)

        # Residual Block 2
        identity = x
        out = self.res2(x)
        x = F.relu(out + identity, inplace=False)

        # Downsampling 2
        identity = self.skip2(x)
        x = self.trans2(x) + identity
        x = F.relu(x, inplace=False)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def get_model(dataset_name):
    """Returns a model appropriate for the dataset."""
    if dataset_name.lower() == "cifar10":
        return CNN_CIFAR10(num_classes=10)
    else:
        raise ValueError("Unsupported dataset. Extend get_model for more datasets.")


def train_local(train_loader, num_local_steps, lr, device, global_model_state_dict, datasetName="cifar10",mu_fedprox=0.01):
    """
    Performs a limited number of local SGD updates before aggregation.
    
    Inputs:
    - vehicle: The vehicle object containing the model and data.
    - train_loader: DataLoader for the vehicle's local dataset.
    - num_local_steps: Number of local SGD steps to perform.
    - lr: Learning rate for the optimizer.
    - device: Device to run the training on (e.g., "cpu" or "cuda").
    - global_model_state_dict: The global model's state_dict for initialization.
    Returns:
    - model.state_dict(): The updated model parameters.
    """
    model = get_model(datasetName)  # Initialize the model
    model.load_state_dict(global_model_state_dict)  # Load the vehicle's model parameters
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    global_model = copy.deepcopy(model) # copy the global model for fedprox penalty


    # Perform exactly num_local_steps updates
    for step, (images, labels) in enumerate(train_loader):
        if step > num_local_steps:  # Stop after required SGD steps
            break

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        if mu_fedprox != 0:
            prox_term = 0.0
            for param, global_param in zip(model.parameters(), global_model.parameters()):
                prox_term += torch.sum((param - global_param) ** 2)
            loss += (mu_fedprox / 2) * prox_term

        loss.backward()
        optimizer.step()
    
    return model.state_dict()

@torch.no_grad() # disable gradient calculation during aggregation
def custom_aggregate_vehicles(vehicleList, epsilon=1e-9):
    """
    Aggregates model parameters from a list of Vehicle objects using a custom formula.


    Returns:
        dict: Aggregated model parameters as a state_dict.
    """
    if not vehicleList:
        raise ValueError("Vehicle list cannot be empty.")

    # --- Initialization ---
    # Use the state_dict from the first vehicle as a template for keys and shapes.
    # Assumes all vehicles have models with the exact same architecture.
    try:
        firstVehicleParams = vehicleList[0].model_state_dict
        if not firstVehicleParams:
             raise ValueError("The first vehicles model_state_dict is empty or None.")
    except AttributeError:
        raise AttributeError("The first vehicle object is missing the 'model_state_dict' attribute.")

    # Initialize the aggregated_params dict with zero tensors matching the structure
    # Place tensors on the same device as the first model's parameters
    aggregated_params = {
        key: torch.zeros_like(param, dtype=torch.float32)
        for key, param in firstVehicleParams.items()
    }
    target_device = next(iter(aggregated_params.values())).device if aggregated_params else torch.device("cpu")


    # --- Weighted Summation ---
    total_weight_sum = 0.0 # Optional: for debugging normalization

    for vehicle in vehicleList:
        # Access attributes directly from the vehicle object
        try:
            params = vehicle.model_state_dict # The state_dict for this vehicle
            d_ratio = vehicle.D_ratio
            u_val = vehicle.u_vt
            P_A = vehicle.P_A
        except AttributeError as e:
            raise AttributeError(f"A Vehicle object in the list is missing required attributes: {e}")

        # --- Calculate Weight (with safety check) ---
        denominator = (u_val * P_A)
        safe_denominator = denominator + epsilon

        if abs(denominator) < epsilon / 10: # Check if the original denominator was effectively zero
             print(f"Warning: Denominator near zero for a vehicle (u_vt={u_val}, P_A={P_A}). Skipping its contribution.")
             continue # Skip to the next vehicle

        weight = d_ratio / safe_denominator  # Compute weighting factor
        total_weight_sum += weight # Optional debug

        # --- Accumulate weighted parameters ---
        for key, param_tensor in params.items():
            if key not in aggregated_params:
                # This error suggests inconsistent model architectures across vehicles
                raise KeyError(f"Parameter key '{key}' found in a vehicle's state_dict but not in the template. "
                               "Model architectures may differ.")

            # Ensure data types and devices are compatible for accumulation
            aggregated_params[key] += weight * param_tensor.to(target_device, dtype=torch.float32)

    # Optional: Print total weight if normalization is expected or for debugging
    # print(f"Debug: Sum of calculated weights: {total_weight_sum_debug}")

    for key, param_tensor in aggregated_params.items():
         aggregated_params[key] /= total_weight_sum  # Normalize by the total weight sum

    return aggregated_params

def get_class_indices(labels_tensor):
    """
    Returns a dictionary mapping class labels to their indices in the dataset.
    
    Inputs:
    - labels_tensor: Full labels tensor of shape (N,)
    
    Outputs:
    - class_indices: Dictionary mapping class labels to their indices.
    """
    classes = torch.unique(labels_tensor)
    class_indices_precomputed = {class_label.item(): torch.nonzero(labels_tensor == class_label).squeeze().tolist() for class_label in classes}
    return class_indices_precomputed

def assign_data_to_vehicle(labels_tensor, vehicle, nof_samples_per_class_per_vehicle, iid=True, class_indices_precomputed=None, number_of_classes=1, number_of_samples=1):
    """
    Assigns data indices to a single vehicle based on IID or non-IID distribution.

    Inputs:
    - labels_tensor: Full labels tensor of shape (N,)
    - vehicle: The vehicle object that needs data assigned
    - nof_samples_per_class: Number of samples per class (used in IID setting)
    - iid: Boolean flag indicating whether to use IID or non-IID distribution
    - class_indices_precomputed: Precomputed dictionary mapping class labels to available indices.
    - number_of_classes: Number of classes to sample from (used in non-IID setting), if iid is True, this value is ignored.
    - number_of_samples: Number of samples to sample from each class (used in non-IID setting), if iid is True, this value is ignored.
    Outputs:
    - Updates the vehicle object with assigned indices in its `data_indices` attribute.
    """


    # # Create a deep copy to avoid modifying the original
    # class_indices = copy.deepcopy(class_indices_precomputed)
    # if class_indices is None:
    #     raise ValueError("class_indices cannot be None. Precompute it using get_class_indices().")
    
    # # Get unique class labels
    # classes = torch.unique(labels_tensor)

    # # Shuffle class indices to ensure randomness
    # for class_label in class_indices:
    #     random.shuffle(class_indices[class_label])

    class_indices = class_indices_precomputed
    
    # Get unique class labels (precompute this if possible)
    classes = torch.unique(labels_tensor)
    
    # Initialize list to store this vehicle's data indices
    vehicle_data_indices = []

    if iid:
        # IID setting: Randomly assign `nof_samples_per_class_per_vehicle` samples per class
        for class_label in classes:
            class_label_item = class_label.item()
            class_data = class_indices[class_label_item]
            selected_indices = np.random.choice(class_data, size=nof_samples_per_class_per_vehicle, replace=False).tolist()
            vehicle_data_indices.extend(selected_indices)
        np.random.shuffle(vehicle_data_indices)
    else:
        # Non-IID setting:
        num_classes = number_of_classes  # Number of classes per client
        while number_of_samples > num_classes * NOF_SAMPLES_PER_CLASS:
            num_classes += 1 # if there is not enough samples, increase the number of classes
        
        selected_classes = random.sample(classes.tolist(), num_classes)  # Randomly select classes
        currentIndices = []
        for class_label in selected_classes:
            currentIndices.extend(class_indices[class_label])  # Add all indices of the selected class

        # vehicle_data_indices = random.sample(currentIndices, number_of_samples)  # Randomly sample indices from the selected classes
        vehicle_data_indices = np.random.choice(currentIndices, size=number_of_samples, replace=False).tolist()

    # Assign indices to the vehicle
    vehicle.data_indices = vehicle_data_indices

# Example usage
# Assuming train_images.pt and train_labels.pt are loaded as tensors:
# data_tensor = torch.load("train_images.pt")  # Size: (N, 3, 32, 32)
# labels_tensor = torch.load("train_labels.pt")  # Size: (N,)

# vehicle_objects = [Vehicle() for _ in range(10)]  # List of vehicle objects to be assigned data
# assign_data_to_vehicles(data_tensor, labels_tensor, vehicle_objects, nof_samples_per_class=100, iid=True)