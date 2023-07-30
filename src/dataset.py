from torchvision import datasets
import torch


class ACIFAR10(datasets.CIFAR10):
    def __init__(self, mean, std, **kwargs):
        super(ACIFAR10, self).__init__(**kwargs)
        self.mean = torch.tensor(mean).reshape(len(mean), 1, 1)
        self.std = torch.tensor(std).reshape(len(std), 1, 1)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        
        return img, target
    
    
class AMNIST(datasets.MNIST):
    def __init__(self, mean, std, **kwargs):
        super(AMNIST, self).__init__(**kwargs)
        self.mean = torch.tensor(mean).reshape(len(mean), 1, 1)
        self.std = torch.tensor(std).reshape(len(std), 1, 1)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        
        return img, target