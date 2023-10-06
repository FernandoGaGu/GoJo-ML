# Script used to test the code from gojo.deepl.loops
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
import sys
import torch
from torch.utils.data import DataLoader
from sklearn import datasets
from sklearn.model_selection import train_test_split


sys.path.append('..')

from gojo import core
from gojo import deepl
from gojo import util


def test_loops():
    # load test dataset (Wine)
    wine_dt = datasets.load_wine()

    # create the target variable. Classification problem 0 vs rest
    # to see the target names you can use wine_dt['target_names']
    y = (wine_dt['target'] == 1).astype(int)
    X = wine_dt['data']

    # standardize input data
    std_X = util.zscoresScaling(X)

    # split Xs and Ys in training and validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        std_X, y, train_size=0.8, random_state=1997, shuffle=True,
        stratify=y)

    # create the dataloaders
    train_dl = DataLoader(
        deepl.loading.TorchDataset(X=X_train, y=y_train),
        batch_size=16, shuffle=True)

    valid_dl = DataLoader(
        deepl.loading.TorchDataset(X=X_valid, y=y_valid),
        batch_size=X_valid.shape[0], shuffle=False)

    # create a basic model
    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], 20),
        torch.nn.ELU(),
        torch.nn.Linear(10, 1),
        torch.nn.Sigmoid()
    )

    output = deepl.fitNeuralNetwork(
        deepl.iterSupervisedEpoch,
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        n_epochs=50,
        loss_fn=torch.nn.BCELoss(),
        optimizer_class=torch.optim.Adam,
        optimizer_params={'lr': 0.001}
    )
    print(output)


def test_ffn_generation():
    model = deepl.ffn.createSimpleFFNModel(
        in_feats=10,
        out_feats=99,
        layer_dims=[100, 60, 20],
        layer_activations=[torch.nn.Tanh(), None, torch.nn.ReLU()],
        layer_dropouts=[0.3, None, 0.1],
        batchnorm=True,
        weights_init=[torch.nn.init.kaiming_uniform_] * 3 + [None],
        output_activation=torch.nn.Sigmoid()
    )
    model = deepl.ffn.createSimpleFFNModel(
        in_feats=100,
        out_feats=1,
        layer_dims=[100, 60, 20],
        layer_activations=torch.nn.ReLU(),
        layer_dropouts=0.3,
        batchnorm=True,
        output_activation=torch.nn.Sigmoid()
    )

    print(model)


if __name__ == '__main__':
    #test_loops()
    test_ffn_generation()
