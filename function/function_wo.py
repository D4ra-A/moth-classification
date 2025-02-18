import torch


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
def get_device() -> torch.device:
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

    for insect, label in data_loader:
        insect = insect.to(device)
        label = label.to(device)

        optimizer.zero_grad()                                           # Clear gradients
        output = model(insect)                                          # Forward pass
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
        for insect, label in data_loader:
            insect = insect.to(device)
            label = label.to(device)

            output = model(insect)                                      # Forward pass
            val_loss += criterion(output, label).item()                 # Compute validation loss

            prediction = output.max(1, keepdim=True)[1]                 # Get the predicted class
            correct += prediction.eq(label.view_as(prediction)).sum().item()    # Count correct predictions

    val_loss /= len(data_loader.dataset)                                # Average loss over the dataset
    val_accuracy = 100.0 * correct / len(data_loader.dataset)           # Calculate accuracy

    return val_loss, val_accuracy
