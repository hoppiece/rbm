import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class RBM(nn.Module):
    """Restricted Boltzmann Machine."""

    def __init__(self, n_vis, n_hid, k=1, *args, **kwargs):
        """Create a RBM."""
        super().__init__()
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

    def gibbs_sampling(self, v):
        for _ in range(self.k - 1):
            h = self.encode(v, binarize=True)  # Binary
            v = self.decode(h, binarize=False)  # Real
        h = self.encode(v, binarize=False)  # Real
        v = self.decode(h.bernoulli(), binarize=False)  # Real
        return v, h

    def forward(self, v):
        v, h = self.gibbs_sampling(v)
        return v

    def visible_to_hidden(self, v):
        # Depreciated
        return self.encode(v, binarize=True)

    def hidden_to_visible(self, h):
        # Depreciated
        return self.decode(h, binarize=True)


class Encoder(nn.Module):
    def __init__(self, n_vis, n_hid):
        super().__init__()
        self.fc1 = nn.Linear(n_vis, n_hid)

    def forward(self, v):
        v = torch.relu(self.fc1(v))
        return v


class Decoder(nn.Module):
    def __init__(self, n_vis, n_hid):
        super().__init__()
        self.fc1 = nn.Linear(n_hid, n_vis)

    def forward(self, h):
        h = torch.relu(self.fc1(h))
        return h


class AutoEncoder(nn.Module):
    def __init__(self, n_vis, n_hid):
        super().__init__()
        self.enc = Encoder(n_vis, n_hid)
        self.dec = Decoder(n_vis, n_hid)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x

    def visible_to_hidden(self, v):
        return self.enc(v)

    def hidden_to_visible(self, h):
        return self.dec(h)
