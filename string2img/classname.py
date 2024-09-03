import torchvision
import torchvision.transforms as transforms

# Load CIFAR-10 dataset (train set here)
transform = transforms.Compose([transforms.ToTensor()])
cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

class_names = cifar10_train.classes


# CIFAR-10 class names
for idx, (_, label) in enumerate(cifar10_train):
    class_name = class_names[label]
    print(f"Image {idx+1:06d}.png: {class_name}")
    if idx == 10:
        break