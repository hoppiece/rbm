import torch
from models import prepare_mnist, prepare_fashion_mnist, prepare_cifar10

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from models import ModelConf, AutoEncoder, RBM, VAE

from logzero import logger

import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [N x M] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X.T, rowvar=True)  # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U, S, V = np.linalg.svd(sigma)
    # U: [M x M] eigenvectors of sigma.
    # S: [M x 1] eigenvalues of sigma.
    # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-10
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))  # [M x M]
    return ZCAMatrix


def gen_model(conf):
    if conf.model_name == "rbm":
        return RBM(conf)
    elif conf.model_name == "autoencoder":
        return AutoEncoder(conf)
    elif conf.model_name == "vae":
        return VAE(conf)
    else:
        raise ValueError("Unknown model is specified.")


def gen_data(conf):
    if conf.dataset == "mnist":
        return prepare_mnist(batch_size=conf.batch_size)
    elif conf.dataset == "fashion_mnist":
        return prepare_fashion_mnist(batch_size=conf.batch_size)
    elif conf.dataset == "cifar10":
        return prepare_cifar10(batch_size=conf.batch_size)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"device: {device}")

    confs = [
        ModelConf(
            batch_size=128,
            n_hid=n_hid,
            n_vis=784,
            lr=0.01,
            n_epoch=n_epoch,
            dataset=dataset,
            optimizer=optimizer,
            model_name=model_name,
            whitening_vis=whitening_vis,
            whitening_learn=whitening_learn,
        )
        for whitening_vis, whitening_learn in [
            (False, False),
            (True, False),
            (True, True),
        ]
        for model_name in ["autoencoder", "vae"]
        for n_hid in [100, 1000, 10]
        for n_epoch in [10, 30]
        for dataset in [
            "mnist",
            "fashion_mnist",
        ]
        for optimizer in [
            "sgd",
            "adam",
            "momentum",
            "rmsprop",
        ]
    ]

    for conf in confs:
        try:
            logger.info(f"Experiment: {str(conf)}")
            conf.device = device
            writer = SummaryWriter()
            train_datasets, train_loader, test_datasets, test_loader = gen_data(conf)

            if conf.whitening_learn or conf.whitening_vis:
                data = train_datasets.data.view(-1, conf.n_vis).numpy()
                P = zca_whitening_matrix(data)

            def generate_vv(model, conf):
                for epoch in tqdm(range(conf.n_epoch)):
                    vv = np.zeros((len(train_datasets), conf.n_hid))

                    for idx, (data, target) in enumerate(
                        train_loader
                    ):  # Sample figures reconstruction.
                        if conf.whitening_vis:
                            batch_size = data.size(0)
                            n_iter = len(train_loader) * epoch
                            batch_white = np.dot(data.view(batch_size, -1).numpy(), P)
                            batch_white = (
                                torch.from_numpy(batch_white.astype(np.float32))
                                .clone()
                                .view(batch_size, -1)
                            )
                            vv[idx * batch_size : (idx + 1) * batch_size, :] = (
                                model.encode(batch_white.to(conf.device))
                                .detach()
                                .cpu()
                                .numpy()
                            )

                        else:
                            batch_size = data.size(0)
                            n_iter = len(train_loader) * epoch
                            vv[idx * batch_size : (idx + 1) * batch_size, :] = (
                                model.encode(data.view(-1, conf.n_vis).to(conf.device))
                                .detach()
                                .cpu()
                                .numpy()
                            )
                    for idx, (data, target) in enumerate(train_loader):
                        if conf.whitening_learn:
                            batch_size = data.size(0)
                            n_step = len(train_loader) * epoch + idx
                            batch_white = np.dot(data.view(batch_size, -1).numpy(), P)
                            batch_white = (
                                torch.from_numpy(batch_white.astype(np.float32))
                                .clone()
                                .view(batch_size, -1)
                            )
                            loss = model.train_step(batch_white.to(conf.device)).cpu()
                            writer.add_scalar(
                                f"{model.__class__.__name__}/{str(conf)}/loss",
                                loss,
                                n_step,
                            )
                            recon_loss = model.recon_loss(batch_white.to(device)).cpu()
                            writer.add_scalar(
                                f"{model.__class__.__name__}/{str(conf)}/recon_loss",
                                recon_loss,
                                n_step,
                            )

                        else:
                            batch_size = data.size(0)
                            n_step = len(train_loader) * epoch + idx
                            loss = model.train_step(data.to(device)).cpu()
                            writer.add_scalar(
                                f"{model.__class__.__name__}/{str(conf)}/loss",
                                loss,
                                n_step,
                            )
                            recon_loss = model.recon_loss(data.to(device)).cpu()
                            writer.add_scalar(
                                f"{model.__class__.__name__}/{str(conf)}/recon_loss",
                                recon_loss,
                                n_step,
                            )
                    yield epoch, n_iter, vv

            model = gen_model(conf).to(device)
            vv_iter = generate_vv(model, conf)

            if not conf.whitening_learn:
                sample_images = next(iter(test_loader))[0]
            else:
                sample_images = next(iter(test_loader))[0]
                sample_images = np.dot(sample_images.numpy().reshape(-1, conf.n_vis), P)
                sample_images = torch.from_numpy(
                    sample_images.astype(np.float32)
                ).clone()

            sample_img = make_grid(sample_images.view(conf.batch_size, 1, 28, 28).data)

            fig, ax = plt.subplots(figsize=(10, 10))
            divider = make_axes_locatable(ax)
            mappable = cm.ScalarMappable(
                cmap=cm.viridis,
                norm=matplotlib.colors.Normalize(vmin=0, vmax=conf.n_epoch),
            )
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(mappable=mappable, cax=cax, label="Epochs")

            xx = np.arange(conf.n_hid) + 1
            log_xx = np.log(xx)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Neurons")
            ax.set_ylabel("Variance explained")

            for n_epoch, n_iter, vv in vv_iter:

                pca = PCA()
                pca.fit(vv)
                yy = pca.explained_variance_
                log_yy = np.log(yy)
                m, c = np.linalg.lstsq(
                    np.vstack([log_xx, np.ones(len(log_xx))]).T, log_yy, rcond=None
                )[0]
                pcm = ax.plot(
                    xx,
                    yy,
                    ".-",
                    label=f"epoch: {n_epoch}, m: {m:.3}",
                    color=cm.viridis(n_epoch / conf.n_epoch),
                )
                # ax.legend()
                ax.grid(visible=True)
                writer.add_figure(
                    f"{model.__class__.__name__}/{str(conf)}/neuron_fireings",
                    fig,
                    global_step=n_iter,
                )

                v_recon = model(sample_images.view(conf.batch_size, -1).to(device))
                sample_recon = make_grid(
                    v_recon.view(conf.batch_size, 1, 28, 28).data
                ).cpu()
                writer.add_image(
                    f"{model.__class__.__name__}/{str(conf)}/sample_input",
                    sample_img,
                    global_step=n_iter,
                )
                writer.add_image(
                    f"{model.__class__.__name__}/{str(conf)}/sample_recon",
                    sample_recon,
                    global_step=n_iter,
                )
            # ax.legend()
            # fig.colorbar(ax)
            ax.grid(visible=True)
            writer.add_figure(
                f"{model.__class__.__name__}/{str(conf)}/neuron_fireings/epoch",
                fig,
                close=True,
            )

            #######################################
            fig, ax = plt.subplots(figsize=(8, 8))
            xx = np.arange(conf.n_hid) + 1
            log_xx = np.log(xx)

            ax.set_xlabel("Neurons")
            ax.set_ylabel("Variance explained")
            ax.set_yscale("log")
            ax.set_xscale("log")
            pca = PCA()
            pca.fit(vv)
            yy = pca.explained_variance_
            log_yy = np.log(yy)
            m, c = np.linalg.lstsq(
                np.vstack([log_xx, np.ones(len(log_xx))]).T, log_yy, rcond=None
            )[0]
            ax.plot(xx, yy, ".-", label=f"epoch: {conf.n_epoch}")
            ax.plot(xx, np.e ** c * xx ** m, label=f"log y= {m:.3} log x + {c:.3}")
            ax.legend()
            ax.grid(visible=True)
            writer.add_figure(
                f"{model.__class__.__name__}/{str(conf)}/neuron_fireings/log", fig
            )

            #######################################
            fig, ax = plt.subplots(figsize=(8, 8))
            xx = np.arange(conf.n_hid) + 1
            log_xx = np.log(xx)

            ax.set_xlabel("Neurons")
            ax.set_ylabel("Variance explained")
            pca = PCA()
            pca.fit(vv)
            yy = pca.explained_variance_
            log_yy = np.log(yy)
            m, c = np.linalg.lstsq(
                np.vstack([log_xx, np.ones(len(log_xx))]).T, log_yy, rcond=None
            )[0]
            ax.plot(xx, yy, ".-", label=f"epoch: {conf.n_epoch}")
            ax.plot(xx, np.e ** c * xx ** m, label=f"log y= {m:.3} log x + {c:.3}")
            ax.legend()
            ax.set_ylim(0, np.max(yy))
            ax.grid(visible=True)
            writer.add_figure(
                f"{model.__class__.__name__}/{str(conf)}/neuron_fireings/not_log", fig
            )

        except Exception as e:
            logger.error(e)
            continue
