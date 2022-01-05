import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms
import pytorch_lightning as pl
from dataclasses import dataclass


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

        return F.mse_loss(v0, vk) / batch

    def recon_loss(self, tensor):
        batch = tensor.size(0)
        v0 = tensor.view(batch, -1)
        vk = self.forward(v0)
        loss = F.mse_loss(v0, vk)
        return loss


class AutoEncoder(nn.Module):
    def __init__(self, conf):
        self.conf = conf
        n_vis = conf.n_vis
        n_hid = conf.n_hid
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_vis, n_hid), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(n_hid, n_vis), nn.ReLU())
        if conf.optimizer == "None":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.conf.lr)
        elif conf.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.conf.lr)
        elif conf.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.conf.lr)

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
        dataset=test_datasets, batch_size=batch_size, shuffle=True
    )

    return train_datasets, train_loader, test_datasets, test_loader


@dataclass
class ModelConf:
    batch_size: int = 128
    n_hid: int = 1000
    n_vis: int = 784
    lr: float = 0.01
    n_epoch: int = 30
    optimizer: str = "None"
