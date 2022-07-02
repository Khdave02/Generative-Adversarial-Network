from torchvision import datasets, transforms
from torch.utils.data import DataLoader

img_size=64
transform = transforms.Compose([
transforms.Resize((img_size,img_size)),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def trainLoader(batch_size):
    train_data = datasets.CIFAR10(
        root="./torch_datasets",
        train=True,
        download=True,
        transform=transform
    )
    train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader

def testLoader(batch_size):
    test_data = datasets.CIFAR10(
        root="./torch_datasets",
        train=False,
        download=True,
        transform=transform
    )  
    test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return test_loader