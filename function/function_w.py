import torch
import torch.nn.functional as F


# Function to generate paths for dataset, tensor storage, model saving, and the moth information table
def get_path(root_path: str) -> dict:
    """
    Generates paths for dataset directories and model saving.

    Parameters:
    root_path (str): The root directory where the dataset and model directories are located.

    Returns:
    dict: A dictionary containing paths for images, tensors, model saving, and moth table.
    """

    path_dict = {}
    path_dict["image_path"] = root_path + "\\dataset"                   # Path to the raw dataset
    path_dict["tensor_path"] = root_path + "\\dataset\\tensor"          # Path to the converted tensor files
    path_dict["save_path"] = root_path + "\\model_trained"              # Path to save the trained model
    path_dict["moth_table"] = root_path + "\\dataset\\MothInfo.xlsx"    # Path to the Excel file containing moth information

    return path_dict


# Function to determine the available device (MPS, CUDA, or CPU)
def get_device():
    """
    Detects and returns the available device for computation.

    Returns:
    torch.device: The available device ('mps', 'cuda', or 'cpu').
    """

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")                                    # Use Metal Performance Shaders (Apple devices)
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")                                   # Use CUDA if a GPU is available
    else:
        DEVICE = torch.device("cpu")                                    # Fallback to CPU if no GPU is available

    print(f"Using PyTorch version: {torch.__version__}\nCurrent device: {DEVICE}")

    return DEVICE


# Training function to perform one epoch of training
def train(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Trains the model for one epoch.

    Parameters:
    model (torch.nn.Module): The model to be trained.
    data_loader (torch.utils.data.DataLoader): DataLoader for training data.
    optimizer (torch.optim.Optimizer): Optimizer for model parameters.
    criterion (torch.nn.Module): Loss function to compute training loss.
    device (torch.device): Device to perform computations on.

    Returns:
    float: Average training loss for the epoch.
    """

    model.train()                                                       # Set the model to training mode
    train_loss = 0                                                      # Initialize training loss

    for ins, env1, env2, env3, env4, label in data_loader:
        ins = ins.to(device)
        env1 = env1.to(device)
        env2 = env2.to(device)
        env3 = env3.to(device)
        env4 = env4.to(device)
        label = label.to(device)

        optimizer.zero_grad()                                           # Clear gradients
        output = model(ins, [env1, env2, env3, env4])                   # Forward pass
        loss = criterion(output, label)                                 # Compute loss
        train_loss += loss.item()                                       # Accumulate training loss
        loss.backward()                                                 # Backpropagation
        optimizer.step()                                                # Update model parameters

    train_loss /= len(data_loader.dataset)                              # Average loss over the dataset

    return train_loss


# Evaluation function to validate the model on the validation dataset
def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple:
    """
    Evaluates the model on the validation dataset.

    Parameters:
    model (torch.nn.Module): The model to be evaluated.
    data_loader (torch.utils.data.DataLoader): DataLoader for validation data.
    criterion (torch.nn.Module): Loss function to compute validation loss.
    device (torch.device): Device to perform computations on.

    Returns:
    tuple: Average validation loss and validation accuracy for the epoch.
    """

    model.eval()                                                        # Set the model to evaluation mode
    val_loss = 0                                                        # Initialize validation loss
    correct = 0                                                         # Count of correct predictions

    with torch.no_grad():                                               # Disable gradient calculation during evaluation
        for ins, env1, env2, env3, env4, label in data_loader:
            ins = ins.to(device)
            env1 = env1.to(device)
            env2 = env2.to(device)
            env3 = env3.to(device)
            env4 = env4.to(device)
            label = label.to(device)

            output = model(ins, [env1, env2, env3, env4])               # Forward pass
            val_loss += criterion(output, label).item()                 # Compute validation loss

            prediction = output.max(1, keepdim=True)[1]                 # Get the predicted class
            correct += prediction.eq(label.view_as(prediction)).sum().item()    # Count correct predictions

    val_loss /= len(data_loader.dataset)                                # Average loss over the dataset
    val_accuracy = 100.0 * correct / len(data_loader.dataset)           # Calculate accuracy

    return val_loss, val_accuracy


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation.
    
    This class computes the class activation map (CAM) for a given input image, highlighting
    the most relevant regions for the model's decision.
    
    Attributes:
        model (torch.nn.Module): The model to be analyzed.
        target_layer (torch.nn.Module): The specific convolutional layer where gradients will be extracted.
        gradients (torch.Tensor): Stores the gradients of the target layer during backpropagation.
        activations (torch.Tensor): Stores the activations (feature maps) of the target layer during the forward pass.
        forward_handle (torch.utils.hooks.RemovableHandle): Forward hook for saving activations.
        backward_handle (torch.utils.hooks.RemovableHandle): Backward hook for saving gradients.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None       # Stores gradients for backpropagation
        self.activations = None     # Stores activations from the forward pass

        # Register hooks to capture forward activations and backward gradients
        self.forward_handle = target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """
        Saves the activation (feature map) from the forward pass.
        
        Parameters:
        module (torch.nn.Module): The layer being hooked.
        input (tuple): Input to the layer (unused).
        output (torch.Tensor): Output from the layer.
        """
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        Saves the gradients from the backward pass.
        
        Parameters:
        module (torch.nn.Module): The layer being hooked.
        grad_input (tuple): Gradients with respect to the input (unused).
        grad_output (tuple): Gradients with respect to the output.
        """
        self.gradients = grad_output[0].detach()
    
    def remove_hooks(self):
        """
        Removes the forward and backward hooks to prevent memory leaks.
        """
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def __call__(self, ins_img, b_imgs, target_class):
        """
        Computes the Grad-CAM heatmap for the given input and target class.
        
        Parameters:
        ins_img (torch.Tensor): The input image tensor (batch size = 1, C, H, W).
        b_imgs (list[torch.Tensor]): Background images used in the model.
        target_class (int): The class index for which Grad-CAM is computed.
        
        Returns:
        torch.Tensor: A heatmap tensor with the same spatial dimensions as the input image.
        """
        self.model.zero_grad()                  # Reset gradients

        output = self.model(ins_img, b_imgs)    # Forward pass
        score = output[0, target_class]         # Get the score for the target class
        score.backward(retain_graph=True)       # Compute gradients with respect to the target class

        # Compute the importance weights by averaging the gradients over spatial dimensions
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Compute the weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU to keep only positive activations
        cam = F.relu(cam)
        
        # Resize CAM to match input image size using bilinear interpolation
        cam = F.interpolate(cam, size=ins_img.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize CAM values to range [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
