import torch
from models import prepare_mnist, prepare_fashion_mnist

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from models import ModelConf, AutoEncoder, RBM

from logzero import logger

import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plfit
from pylab import *


def gen_model(conf):
    if conf.model_name == "rbm":
        return RBM(conf)
    elif conf.model_name == "autoencoder":
        return AutoEncoder(conf)
    else:
        raise


def gen_data(conf):
    if conf.dataset == "mnist":
        return prepare_mnist(batch_size=conf.batch_size)
    elif conf.dataset == "fashion_mnist":
        return prepare_fashion_mnist(batch_size=conf.batch_size)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"device: {device}")

    confs = [
        ModelConf(
            batch_size=128,
            n_hid=1000,
            n_vis=784,
            lr=0.01,
            n_epoch=10,
            dataset="mnist",
            model_name="rbm",
        )
    ]

    for conf in confs:
        logger.info(f"Experiment: {str(conf)}")
        writer = SummaryWriter()
        train_datasets, train_loader, test_datasets, test_loader = gen_data(conf)

        def generate_vv(model, conf, fast=False):
            for epoch in tqdm(range(conf.n_epoch)):
                vv = np.zeros((len(train_datasets), conf.n_hid))
                if fast:
                    for idx, (data, target) in enumerate(train_loader):
                        batch_size = data.size(0)
                        n_iter = len(train_loader) * epoch + idx
                        recon_loss = model.train_step(data.to(device)).to("cpu")
                        writer.add_scalar(
                            f"{model.__class__.__name__}/{str(conf)}/recon_loss",
                            recon_loss,
                            n_iter,
                        )
                        vv[idx * batch_size : (idx + 1) * batch_size, :] = (
                            model.encode(data.view(-1, conf.n_vis).to(device))
                            .detach()
                            .cpu()
                            .numpy()
                        )
                else:
                    for idx, (data, target) in enumerate(train_loader):
                        batch_size = data.size(0)
                        n_iter = len(train_loader) * epoch
                        vv[idx * batch_size : (idx + 1) * batch_size, :] = (
                            model.encode(data.view(-1, conf.n_vis).to(device))
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    for idx, (data, target) in enumerate(train_loader):
                        batch_size = data.size(0)
                        n_step = len(train_loader) * epoch + idx
                        recon_loss = model.train_step(data.to(device)).cpu()
                        writer.add_scalar(
                            f"{model.__class__.__name__}/{str(conf)}/recon_loss",
                            recon_loss,
                            n_step,
                        )
                yield epoch, n_iter, vv

        model = gen_model(conf).to(device)
        vv_iter = generate_vv(model, conf)

        sample_images = next(iter(test_loader))[0]
        sample_img = make_grid(sample_images.view(conf.batch_size, 1, 28, 28).data)

        fig, ax = plt.subplots(figsize=(8, 8))
        divider = make_axes_locatable(ax)
        mappable = cm.ScalarMappable(
            cmap=cm.viridis, norm=matplotlib.colors.Normalize(vmin=0, vmax=conf.n_epoch)
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

        #######################################
        myplfit = plfit.plfit(yy)
        myplfit.discrete_best_alpha()
        figure(1)
        myplfit.plotpdf()
        writer.add_figure(f"{model.__class__.__name__}/{str(conf)}/plotpdf", figure(1))

        figure(2)
        myplfit.plotcdf()

        figure(3)
        myplfit.alphavsks()
