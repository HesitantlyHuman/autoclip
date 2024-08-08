"""
Adapted from the pytorch mnist example found at https://github.com/pytorch/examples/blob/main/mnist/main.py
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from autoclip.torch import QuantileClip


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device = torch.device("cuda"),
):
    model.train()
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()


def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cuda"),
) -> float:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss


def run_trial(
    epochs: int,
    learning_rate: float,
    regularization: bool,
    quantile: float,
    use_clipping: bool,
):
    config = {
        "num_epochs": epochs,
        "max_learning_rate": learning_rate,
        "weight_decay": 0.05,
    }
    train_kwargs = {"batch_size": 64}
    test_kwargs = {"batch_size": 1000}
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_kwargs = {"num_workers": 12, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    else:
        device = torch.device("cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["max_learning_rate"],
        weight_decay=config["weight_decay"],
    )
    if use_clipping:
        optimizer = QuantileClip.as_optimizer(
            optimizer=optimizer,
            quantile=quantile,
            history_length=1000,
            lr_regularize=regularization,
        )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["max_learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=config["num_epochs"],
    )

    best_test_loss = torch.inf
    for _ in range(1, config["num_epochs"] + 1):
        train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            device=device,
        )
        test_loss = test(model=model, test_loader=test_loader, device=device)
        if test_loss < best_test_loss:
            best_test_loss = test_loss

    return best_test_loss


if __name__ == "__main__":
    import csv
    from tqdm import tqdm

    use_clipping = False
    overall_progress = tqdm(total=5 * 30, desc="All Trials", smoothing=0.95)

    with open("experiments/results/mnist.csv", "a") as f:
        writer = csv.writer(f)
        epoch_progress_bar = tqdm([10], desc="Epochs _")
        for epochs in epoch_progress_bar:
            epoch_progress_bar.set_description(f"Epochs {epochs}")
            lr_progress_bar = tqdm(
                [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], desc="Learning Rate _"
            )
            for learning_rate in lr_progress_bar:
                lr_progress_bar.set_description(f"Learning Rate {learning_rate}")
                for _ in tqdm(range(30), desc="Trials"):
                    best_loss = run_trial(epochs, learning_rate, False, 1.0, False)
                    writer.writerow(
                        [epochs, learning_rate, False, 1.0, best_loss, False]
                    )
                    overall_progress.update(1)
                # regularize_progress_bar = tqdm([True, False], desc="Regularization _")
                # for regularize in regularize_progress_bar:
                #     regularize_progress_bar.set_description(
                #         f"Regularization {regularize}"
                #     )
                #     quantile_progress_bar = tqdm(
                #         [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0], desc="Quantile _"
                #     )
                #     for quantile in quantile_progress_bar:
                #         quantile_progress_bar.set_description(f"Quantile {quantile}")
                #         for trial in tqdm(range(5), desc="Trials"):
                #             best_loss = run_trial(
                #                 epochs, learning_rate, regularize, quantile
                #             )
                #             writer.writerow(
                #                 [epochs, learning_rate, regularize, quantile, best_loss]
                #             )
                #             overall_progress.update(1)
