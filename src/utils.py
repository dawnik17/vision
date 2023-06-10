import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += (pred.argmax(-1) == target).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f"Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
        )

    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))


def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            pred = model(data)
            test_loss += criterion(
                pred, target, reduction="sum"
            ).item()  # sum up batch loss

            correct += (pred.argmax(-1) == target).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc.append(100.0 * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_loss


def plot_stats():
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


# plot correct/incorrect predictions
class PlotOutput:
    def __init__(self, device: str):
        self.result = None
        self.device = device
        self.title = (
            "Actual Label: {target} \n Predicted Label: {prediction} \n Score: {score}"
        )

    def reset(self):
        self.result = {"images": [], "target": [], "prediction": [], "confidence": []}

    def run_prediction(self, model, data_loader, ptype="incorrect"):
        self.reset()
        model.eval()

        with torch.no_grad():
            for (data, target) in tqdm(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = model(data)
                prediction = F.softmax(prediction, dim=-1)

                # filter out correct/incorrect results
                score, values = prediction.max(-1)
                idx = values != target if ptype == "incorrect" else values == target
                idx = idx.nonzero(as_tuple=True)[0].tolist()

                self.result["images"].extend(list(data[idx].cpu()))
                self.result["prediction"].extend(values[idx].tolist())
                self.result["target"].extend(target[idx].tolist())
                self.result["confidence"].extend(score[idx].tolist())

    def plot(self, n: int, reverse: bool = True):
        grid_size = (math.ceil(n / 4), 4)
        fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 8))
        fig.tight_layout()

        sorted_indices = np.argsort(self.result["confidence"])

        if reverse:
            sorted_indices = sorted_indices[::-1]

        for i, ax in enumerate(axs.flat):
            j = sorted_indices[i]
            ax.imshow(self.result["images"][j].squeeze(0), cmap="gray")

            ititle = self.title.format(
                target=self.result["target"][j],
                prediction=self.result["prediction"][j],
                score=round(self.result["confidence"][j], 2),
            )
            ax.set_title(ititle, fontsize=7)
            ax.axis("off")

        plt.subplots_adjust(hspace=1)
        plt.rcParams["font.size"] = 7
        plt.show()
