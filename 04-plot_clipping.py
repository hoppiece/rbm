import torch
from models import prepare_mnist, prepare_fashion_mnist, prepare_sub_mnist

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from models import ModelConf

from logzero import logger

from models import AutoEncoder, RBM, MLPAutoEncoder, VAE

import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == "__main__":
    writer = SummaryWriter("runs/autoencoder_clipping")

    confs0 = [
        ModelConf(
            optimizer="momentum",
            dataset=dataset,
            model_name="autoencoder",
            whitening_vis=True,
        )
        for dataset in ["mnist", "fashion_mnist"]
    ]
    confs1 = [
        ModelConf(
            optimizer="rmsprop",
            dataset=dataset,
            model_name="autoencoder",
            whitening_vis=whitening_vis,
        )
        for whitening_vis in [False, True]
        for dataset in ["mnist", "fashion_mnist"]
    ]

    confs = confs0 + confs1
    y_clips = [
        (10 ** (-8), 10 ** (-2.5)),
        (10 ** (-8), 10 ** (-2.5)),
        (10 ** (-3), 10 ** 4),
        (10 ** (-3), 10 ** 3),
        (10 ** (-6), 1),
        (10 ** (-6), 1),
    ]
    for conf_idx, conf in enumerate(confs):
        logger.info(f"Conf {conf_idx+1}/{len(confs)}: {str(conf)}")

        try:
            device = (
                torch.device("cuda:1")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            conf.device = device

            if conf.dataset == "mnist":
                (
                    train_datasets,
                    train_loader,
                    test_datasets,
                    test_loader,
                ) = prepare_mnist(batch_size=conf.batch_size)
            if conf.dataset == "fashion_mnist":
                (
                    train_datasets,
                    train_loader,
                    test_datasets,
                    test_loader,
                ) = prepare_fashion_mnist(batch_size=conf.batch_size)

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
                ZCAMatrix = np.dot(
                    U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T)
                )  # [M x M]
                return ZCAMatrix

            data = train_datasets.data.view(-1, conf.n_vis).numpy()
            P = zca_whitening_matrix(data)

            plt.imshow(data[0].reshape(28, 28))

            plt.imshow(np.dot(data, P).reshape((-1, 28, 28))[0])
            data_white = np.dot(data, P).reshape((-1, conf.n_vis))
            data_white.shape

            def generate_vv(model, conf, fast=False):
                for epoch in tqdm(range(conf.n_epoch)):
                    vv = np.zeros((len(train_datasets), conf.n_hid))
                    if fast:
                        for idx, (data, target) in enumerate(train_loader):
                            batch_size = data.size(0)
                            n_iter = len(train_loader) * epoch + idx
                            recon_loss = model.train_step(data.to(conf.device)).to(
                                "cpu"
                            )
                            writer.add_scalar(
                                f"{model.__class__.__name__}/{str(conf)}/recon_loss",
                                recon_loss,
                                n_iter,
                            )
                            vv[idx * batch_size : (idx + 1) * batch_size, :] = (
                                model.encode(data.view(-1, conf.n_vis).to(conf.device))
                                .detach()
                                .cpu()
                                .numpy()
                            )
                    else:
                        for idx, (data, target) in enumerate(train_loader):
                            if conf.whitening_vis:
                                batch_size = data.size(0)
                                n_iter = len(train_loader) * epoch
                                batch_white = np.dot(
                                    data.view(batch_size, -1).numpy(), P
                                )
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
                                    model.encode(
                                        data.view(-1, conf.n_vis).to(conf.device)
                                    )
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                        for idx, (data, target) in enumerate(train_loader):
                            if conf.whitening_learn:
                                batch_size = data.size(0)
                                n_step = len(train_loader) * epoch + idx
                                batch_white = np.dot(
                                    data.view(batch_size, -1).numpy(), P
                                )
                                batch_white = (
                                    torch.from_numpy(batch_white.astype(np.float32))
                                    .clone()
                                    .view(batch_size, -1)
                                )
                                loss = model.train_step(
                                    batch_white.to(conf.device)
                                ).cpu()
                                writer.add_scalar(
                                    f"{model.__class__.__name__}/{str(conf)}/loss",
                                    loss,
                                    n_step,
                                )
                                recon_loss = model.recon_loss(
                                    batch_white.to(device)
                                ).cpu()
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

            if conf.model_name == "rbm":
                model = RBM(conf=conf).to(conf.device)
            elif conf.model_name == "autoencoder":
                model = AutoEncoder(conf).to(conf.device)
            elif conf.model_name == "vae":
                model = VAE(conf).to(conf.device)
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
            fig.colorbar(mappable=mappable, cax=cax, label="Epoch")

            xx = np.arange(conf.n_hid) + 1
            log_xx = np.log(xx)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("N")
            ax.set_ylabel("Variance")
            ax.grid(visible=True)
            ax.set_ylim(y_clips[conf_idx])
            labels = []

            for n_epoch, n_iter, vv in vv_iter:

                pca = PCA()
                pca.fit(vv)
                yy = pca.explained_variance_
                log_yy = np.log(yy)
                m, c = np.linalg.lstsq(
                    np.vstack([log_xx, np.ones(len(log_xx))]).T, log_yy, rcond=None
                )[0]
                pcm = ax.plot(xx, yy, ".-", color=cm.viridis(n_epoch / conf.n_epoch))

                label = f"{str(conf)}/neuron_fireings"
                writer.add_figure(label, fig, global_step=n_epoch)
                labels.append(label)

                v_recon = model(sample_images.view(conf.batch_size, -1).to(device))
                sample_recon = make_grid(
                    v_recon.view(conf.batch_size, 1, 28, 28).data
                ).cpu()
                writer.add_image(
                    f"{str(conf)}/sample_input", sample_img, global_step=n_epoch
                )
                writer.add_image(
                    f"{str(conf)}/sample_recon", sample_recon, global_step=n_epoch
                )

                cum_var = pca.explained_variance_ratio_.cumsum()
                ef_dim = np.argmax(np.where(cum_var < 0.99, cum_var, 0))
                writer.add_scalar(
                    f"{str(conf)}/effective_dim", ef_dim, global_step=n_epoch
                )

            cum_var = pca.explained_variance_ratio_.cumsum()
            th = 0.99
            x_max = np.argmax(np.where(cum_var < th, cum_var, 0))
            print("x_max", x_max)
            x_th = np.arange(x_max) + 1
            y_th = pca.explained_variance_[:x_max]
            log_x_th = np.log(x_th)
            log_y_th = np.log(y_th)
            m, c = np.linalg.lstsq(
                np.vstack([log_x_th, np.ones(len(log_x_th))]).T, log_y_th, rcond=None
            )[0]

            xx = np.arange(conf.n_hid) + 1
            ax.plot(
                xx,
                np.e ** c * xx ** m,
                "--",
                c="magenta",
                label=f"Fitting to 99% of variance",
            )
            ax.scatter(
                [x_max],
                [np.e ** c * x_max ** m],
                c="red",
                marker="o",
                linewidths=3,
                label=f"a, n_max: {m:.3}, {x_max}",
                zorder=2.5,
            )
            ax.grid(visible=True)
            ax.legend()

            writer.add_figure(f"{str(conf)}/neuron_fireings/finish", fig)
        except Exception as e:
            logger.error(e)
