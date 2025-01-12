from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def create_mnist_dataset(data_path, batch_size, **kwargs):
    # Get optional parameters with default values
    train = kwargs.get("train", True)
    download = kwargs.get("download", True)

    # Create the MNIST dataset with specified transformations
    dataset = MNIST(root=data_path, train=train, download=download, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally with a probability of 0.5
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize images with mean and std deviation of 0.5
    ]))

    # Set DataLoader parameters with default values
    loader_params = dict(
        shuffle=kwargs.get("shuffle", True),  # Shuffle the data at every epoch
        drop_last=kwargs.get("drop_last", True),  # Drop the last incomplete batch
        pin_memory=kwargs.get("pin_memory", True),  # Copy tensors into CUDA pinned memory
        num_workers=kwargs.get("num_workers", 4),  # Number of subprocesses to use for data loading
    )

    # Create a DataLoader with the dataset and specified parameters
    dataloader = DataLoader(dataset, batch_size=batch_size, **loader_params)

    return dataloader  # Return the DataLoader
