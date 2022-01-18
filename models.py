import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms
import pytorch_lightning as pl
from dataclasses import dataclass
import dataclasses

import numpy as np
import math
from collections import OrderedDict


class RBM(nn.Module):
    """Restricted Boltzmann Machine."""

    def __init__(self, conf, k=1, *args, **kwargs):
        """Create a RBM."""
        super().__init__()
        self.conf = conf
        n_vis = conf.n_vis
        n_hid = conf.n_hid
        self.a = nn.Parameter(torch.randn(1, n_vis))  # Bias for visible units
        self.b = nn.Parameter(torch.randn(1, n_hid))  # Bias for hidden units
        self.W = nn.Parameter(torch.randn(n_hid, n_vis))  # Weight parameter
        self.k = k  # The number of iteration in CD-k method

    def encode(self, v, binarize=False):
        """Conditional sampling a hidden variable given a visible variable.
        Args:
            v (Tensor): The visible variable.
            binarize (bool): Applying Bernoulli sampling. Default, False.
        Returns:
            Tensor: The hidden variable.
        """
        p = torch.sigmoid(F.linear(v, self.W, self.b))
        return p.bernoulli() if binarize else p

    def decode(self, h, binarize=False):
        r"""Conditional sampling a visible variable given a hidden variable.
        Args:
            h (Tendor): The hidden variable.
            binarize (bool): Applying Bernoulli sampling. Default, False.
        Returns:
            Tensor: The visible variable.
        """
        p = torch.sigmoid(F.linear(h, self.W.t(), self.a))
        return p.bernoulli() if binarize else p

    def free_energy(self, v):
        r"""Free energy function.
        .. math::
            \begin{align}
                F(x) &= -\log \sum_h \exp (-E(x, h)) \\
                &= -a^\top x - \sum_j \log (1 + \exp(W^{\top}_jx + b_j))\,.
            \end{align}
        Args:
            v (Tensor): The visible variable.
        Returns:
            FloatTensor: The free energy value.
        """
        v_term = torch.matmul(v, self.a.t())
        w_x_h = F.linear(v, self.W, self.b)
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return torch.mean(-h_term - v_term)

    def gibbs_sampling(self, v, k):
        """
        Approximation of the Gibbs sampling of visible & hidden units
        using CD-k method.
        """
        for _ in range(k):
            h = self.encode(v, binarize=True)  # Binary
            v = self.decode(h, binarize=False)  # Real
        h = self.encode(v, binarize=False)  # Real
        return v, h

    def forward(self, v):
        v, h = self.gibbs_sampling(v, self.k)
        return v

    def visible_to_hidden(self, v):
        # Depreciated
        return self.encode(v, binarize=True)

    def hidden_to_visible(self, h):
        # Depreciated
        return self.decode(h, binarize=True)

    def train_step(self, tensor, *args, **kwargs):
        """
        Updating a internal parameters from input batch tensor
        Args:
            tensor (torch.Tensor): Input picture tensor. The shape must be [batch_size, [vectors]]
            lr (float): Learning rate
        """
        lr = self.conf.lr
        batch = tensor.size(0)
        v0 = tensor.view(batch, -1)
        h0 = self.encode(v0, binarize=False)
        vk, hk = self.gibbs_sampling(v0, self.k)

        eta = lr / batch
        self.W.data += eta * (torch.mm(h0.t(), v0) - torch.mm(hk.t(), vk))
        self.a.data += eta * torch.sum(v0 - vk, 0)
        self.b.data += eta * torch.sum(h0 - hk, 0)

        return F.mse_loss(v0, vk)

    def recon_loss(self, tensor):
        batch = tensor.size(0)
        v0 = tensor.view(batch, -1)
        vk = self.forward(v0)
        loss = F.mse_loss(v0, vk)
        return loss


class AutoEncoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        n_vis = conf.n_vis
        n_hid = conf.n_hid
        self.encoder = nn.Sequential(nn.Linear(n_vis, n_hid), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(n_hid, n_vis), nn.Sigmoid())
        if conf.optimizer == "None":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.conf.lr)
        elif conf.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.conf.lr)
        elif conf.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.conf.lr)
        elif conf.optimizer == "momentum":
            self.optimizer = torch.optim.SGD(
                self.parameters(), lr=self.conf.lr, momentum=0.9, dampening=0.9
            )
        elif conf.optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.conf.lr)
        else:
            raise ValueError("Failed to initialize of optimizer")

    def encode(self, v):
        return self.encoder(v)

    def decode(self, h):
        return self.decoder(h)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def visible_to_hidden(self, v):
        return self.encode(v)

    def hidden_to_visible(self, h):
        return self.decode(h)

    def recon_loss(self, tensor):
        batch = tensor.size(0)
        v = tensor.view(batch, -1)
        recon = self.forward(v)
        loss = F.mse_loss(v, recon)
        return loss

    def train_step(self, tensor, *args, **kwargs):
        loss = self.recon_loss(tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


class AutoEncoderCifar10(AutoEncoder):
    def __init__(self, conf):
        super().__init__(conf)
        n_vis = conf.n_vis
        n_hid = conf.n_vis // 8
        self.encoder = nn.Sequential(nn.Linear(n_vis, n_vis // 2), nn.ReLU())


class MLPAutoEncoder(AutoEncoder):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf
        L = 3

        n_vis = conf.n_vis
        n_hid = conf.n_hid
        n_mids = [
            n_vis + round((n_hid - n_vis) / L * i) for i in range(1, L)
        ]  # Interpolate the number of middle layers between vis and hid.

        self.encoder = nn.Sequential(
            nn.Linear(n_vis, n_mids[0]),
            nn.ReLU(),
            nn.Linear(n_mids[0], n_mids[1]),
            nn.ReLU(),
            nn.Linear(n_mids[1], n_hid),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_hid, n_mids[1]),
            nn.ReLU(),
            nn.Linear(n_mids[1], n_mids[0]),
            nn.ReLU(),
            nn.Linear(n_mids[0], n_vis),
        )
        if conf.optimizer == "None":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.conf.lr)
        elif conf.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.conf.lr)
        elif conf.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.conf.lr)
        elif conf.optimizer == "momentum":
            self.optimizer = torch.optim.SGD(
                self.parameters(), lr=self.conf.lr, momentum=0.9, dampening=0.9
            )
        elif conf.optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.conf.lr)


class VAE(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        n_vis = conf.n_vis
        n_hid = conf.n_hid
        self.encoder_mean = nn.Sequential(nn.Linear(n_vis, n_hid), nn.ReLU())
        # https://stackoverflow.com/questions/49634488/keras-variational-autoencoder-nan-loss
        self.encoder_var = nn.Sequential(
            OrderedDict(
                [
                    ("enc_var_fc1", nn.Linear(n_vis, n_hid)),
                    ("enc_var_relu1", nn.ReLU()),
                ]
            )
        )
        torch.nn.init.zeros_(self.encoder_var.enc_var_fc1.weight)
        self.decoder = nn.Sequential(nn.Linear(n_hid, n_vis), nn.Sigmoid())

        if conf.optimizer == "None":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.conf.lr)
        elif conf.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.conf.lr)
        elif conf.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.conf.lr)
        elif conf.optimizer == "momentum":
            self.optimizer = torch.optim.SGD(
                self.parameters(), lr=self.conf.lr, momentum=0.9, dampening=0.9
            )
        elif conf.optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.conf.lr)

    def enc_param(self, v):
        mean, log_var = self.encoder_mean(v), self.encoder_var(v)
        return mean, log_var

    def sample_h(self, mean, log_var):
        epsilon = torch.randn(mean.shape).to(self.conf.device)
        # Next code sometimes makes NaN for large log_var, so weights of self.encoder_var are initialized to zero.
        h = mean + epsilon * torch.exp(0.5 * log_var)
        return h

    def encode(self, v):
        mean, var = self.enc_param(v)
        h = self.sample_h(mean, var)
        return h

    def decode(self, h):
        return self.decoder(h)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def recon_loss(self, tensor):
        batch = tensor.size(0)
        v = tensor.view(batch, -1)
        recon = self.forward(v)
        loss = F.mse_loss(v, recon)
        return loss

    def loss(self, tensor):
        batch = tensor.size(0)
        x = tensor.view(batch, -1)
        mean, log_var = self.enc_param(x)
        delta = 1e-8
        KL = 0.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var))

        z = self.sample_h(mean, log_var)
        x_hat = self.decode(z)
        reconstruction = torch.sum(
            x * torch.log(x_hat + delta) + (1 - x) * torch.log(1 - x_hat + delta)
        )
        lower_bound = KL + reconstruction
        return -lower_bound

    def train_step(self, tensor, *args, **kwargs):
        loss = self.loss(tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


def prepare_mnist(batch_size=128):
    train_datasets = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0, 1)]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_datasets,
        batch_size=batch_size,
        shuffle=True,
    )
    test_datasets = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_datasets, batch_size=batch_size
    )

    return train_datasets, train_loader, test_datasets, test_loader


def prepare_sub_mnist(batch_size=128, n_set=5):
    train_datasets = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(1, 0)]
        ),
    )
    train_mask = train_datasets.targets == 5
    train_datasets.data = train_datasets.data[train_mask]
    train_datasets.targets = train_datasets.targets[train_mask]

    test_datasets = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_mask = test_datasets.targets < n_set
    test_datasets.data = test_datasets.data[test_mask]
    test_datasets.targets = test_datasets.targets[test_mask]

    train_loader = torch.utils.data.DataLoader(
        dataset=train_datasets,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_datasets, batch_size=batch_size
    )
    return train_datasets, train_loader, test_datasets, test_loader


def prepare_fashion_mnist(batch_size=128):
    train_datasets = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0, 1)]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_datasets,
        batch_size=batch_size,
        shuffle=True,
    )
    test_datasets = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_datasets, batch_size=batch_size
    )

    return train_datasets, train_loader, test_datasets, test_loader


def prepare_cifar10(batch_size=128):
    train_datasets = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Grayscale()]),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_datasets,
        batch_size=batch_size,
        shuffle=True,
    )
    test_datasets = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Grayscale()]),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_datasets, batch_size=batch_size
    )
    return train_datasets, train_loader, test_datasets, test_loader


@dataclass
class ModelConf:
    batch_size: int = 128
    n_hid: int = 1000
    n_vis: int = 784
    lr: float = 0.01
    n_epoch: int = 30
    optimizer: str = ""
    dataset: str = ""
    model_name: str = ""
    whitening_vis: bool = False
    whitening_learn: bool = False

    def __str__(self):
        d = dataclasses.asdict(self)
        tags = "/"
        for key, val in d.items():
            tags += f"{key}={str(val)}/"
        return tags


@dataclass
class MNISTConf(ModelConf):
    pass
