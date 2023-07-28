import math
from typing import List
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm


class Train:
    def __init__(self, model, trainloader, optimizer, criterion, scheduler=None):
        self.device = self.get_device()
        self.model = model.to(self.device)

        self.trainloader = trainloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = (
            scheduler if scheduler.__class__.__name__.lower() == "onecyclelr" else None
        )

        self.train_losses = []
        self.train_accuracy = []

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self):
        self.model.train()

        pbar = tqdm(self.trainloader)
        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Predict
            pred = self.model(data)

            # Calculate loss
            loss = self.criterion(pred, target)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            correct += (pred.argmax(-1) == target).sum().item()
            processed += len(target)

            # Display progress
            pbar.set_description(
                desc=f"Train: Loss={train_loss / (batch_idx + 1):0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}"
            )

        # Record training statistics
        self.train_losses.append(train_loss / len(self.trainloader))
        self.train_accuracy.append(100 * correct / processed)

    def plot_stats(self):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(self.train_losses, label="Training Loss")
        axs[0].set_title("Training Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        axs[1].plot(self.train_accuracy, label="Training Accuracy")
        axs[1].set_title("Training Accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy (%)")
        axs[1].legend()

        plt.tight_layout()
        plt.show()


class Test:
    def __init__(self, model, testloader, criterion):
        self.device = self.get_device()
        self.model = model.to(self.device)

        self.testloader = testloader
        self.criterion = criterion

        self.test_losses = []
        self.test_accuracy = []

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total_samples = len(self.testloader.dataset)

        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                pred = self.model(data)

                test_loss += self.criterion(pred, target, reduction="sum").item()
                correct += (pred.argmax(-1) == target).sum().item()

        # Calculate and record test statistics
        test_loss /= total_samples

        self.test_losses.append(test_loss)
        self.test_accuracy.append(100.0 * correct / total_samples)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss, correct, total_samples, 100.0 * correct / total_samples
            )
        )

    def plot_stats(self):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(self.test_losses, label="Testing Loss")
        axs[0].set_title("Testing Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        axs[1].plot(self.test_accuracy, label="Testing Accuracy")
        axs[1].set_title("Testing Accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy (%)")
        axs[1].legend()

        plt.tight_layout()
        plt.show()


class TrainTest:
    def __init__(
        self,
        model,
        trainloader,
        testloader,
        optimizer,
        criterion,
        scheduler,
        target_layers=None,
    ):
        self.train = Train(model, trainloader, optimizer, criterion, scheduler)
        self.test = Test(model, testloader, criterion)

        self.scheduler_per_epoch = scheduler.__class__.__name__.lower() != "onecyclelr"
        self.scheduler = scheduler

        self.cam = None
        self.target_layers = target_layers
        self.cam_results = dict()

    def __call__(self, epochs, cam=False, image_idx=None):
        if cam and image_idx is None:
            image_idx = [15]

        for epoch in range(epochs):
            self.train()
            self.test()

            if cam:
                self.grad_cam(image_idx)

            if self.scheduler_per_epoch:
                self.scheduler.step()

    def plot(self):
        self.train.plot_stats()
        self.test.plot_stats()

    def grad_cam(self, image_idx):
        if self.cam is None:
            self.cam = CAM(
                model=self.train.model,
                target_layers=self.target_layers,
                device=self.train.device,
            )

        dataset = self.test.testloader.dataset
        mean = dataset.mean
        std = dataset.std

        for idx in image_idx:
            image, label = dataset[idx]

            input_tensor = (image * std + mean).unsqueeze(0)
            visualization = self.cam(input_tensor, label)

            if idx not in self.cam_results:
                self.cam_results[idx] = [
                    label,
                    (input_tensor[0].permute(1, 2, 0) * 255).numpy().astype("uint8"),
                    visualization,
                ]

            else:
                self.cam_results[idx].append(visualization)

    def save_grad_cam_gif(self, image_idx, output_directory="./cam_output/"):
        # save CAM output as gifs
        for idx in image_idx:
            images = self.cam_results[idx]
            label, images = (
                self.train.trainloader.dataset.classes[images[0]],
                images[1:],
            )

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            image_path = os.path.join(output_directory, f"{label}.gif")
            imageio.mimsave(image_path, images, duration=10 * len(images))


class CAM:
    def __init__(self, model, target_layers, device):
        self.cam = GradCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=device == "cuda",
        )

    def __call__(self, input_tensor, label):
        targets = [ClassifierOutputTarget(label)]

        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=True,
            eigen_smooth=True,
        )

        input_tensor = input_tensor.permute(0, 2, 3, 1).numpy()
        return show_cam_on_image(input_tensor, grayscale_cam[0, :], use_rgb=True)[0]


# plot correct/incorrect predictions
class PlotOutput:
    def __init__(self, model, target_layers, device: str):
        self.mean = None
        self.std = None
        self.cam = None
        self.result = None

        self.device = device
        self.title = (
            "Actual Label: {target} \n Predicted Label: {prediction} \n Score: {score}"
        )

        self.model = model
        self.target_layers = target_layers

    def reset(self):
        self.result = {"images": [], "target": [], "prediction": [], "confidence": []}

    def denormalise(self, tensor):
        return tensor * self.std + self.mean

    def run_prediction(self, data_loader, ptype="incorrect"):
        self.reset()
        self.model.eval()

        self.mean = data_loader.dataset.mean
        self.std = data_loader.dataset.std

        with torch.no_grad():
            for (data, target) in tqdm(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                prediction = F.softmax(prediction, dim=-1)

                # filter out correct/incorrect results
                score, values = prediction.max(-1)
                idx = values != target if ptype == "incorrect" else values == target
                idx = idx.nonzero(as_tuple=True)[0].tolist()

                self.result["images"].extend(list(data[idx].cpu()))
                self.result["prediction"].extend(values[idx].tolist())
                self.result["target"].extend(target[idx].tolist())
                self.result["confidence"].extend(score[idx].tolist())

    def plot(
        self, n: int, class_list: List, reverse: bool = True, grad_cam: bool = False
    ):
        if grad_cam and self.cam is None:
            self.cam = CAM(
                model=self.model, target_layers=self.target_layers, device=self.device
            )

        cmap = "gray" if self.mean.shape[0] == 1 else None

        grid_size = (math.ceil(n / 4), 4)
        fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 8))
        fig.tight_layout()

        sorted_indices = np.argsort(self.result["confidence"])

        if reverse:
            sorted_indices = sorted_indices[::-1]

        for i, ax in enumerate(axs.flat):
            j = sorted_indices[i]

            image = self.result["images"][j]
            image = self.denormalise(image)
            image = image.clip(0, 1)

            if not grad_cam:
                image = image.squeeze(0) if cmap == "gray" else image.permute(1, 2, 0)

            else:
                # image.unsqueeze(0) shape [1, 3, 32, 32]
                image = self.cam(
                    input_tensor=image.unsqueeze(0), label=self.result["prediction"][j]
                )

            ax.imshow(image, cmap=cmap)

            ititle = self.title.format(
                target=class_list[self.result["target"][j]],
                prediction=class_list[self.result["prediction"][j]],
                score=round(self.result["confidence"][j], 2),
            )
            ax.set_title(ititle, fontsize=7)
            ax.axis("off")

        plt.subplots_adjust(hspace=1)
        plt.rcParams["font.size"] = 7
        plt.show()
