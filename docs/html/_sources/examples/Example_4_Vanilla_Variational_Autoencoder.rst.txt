Vanilla Variational Autoencoder (vVAE)
======================================

.. code:: python

    import torch
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    import umap
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.manifold import TSNE
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from tqdm import tqdm
    
    # GOJO libraries
    from gojo import deepl
    from gojo import plotting
    from gojo import core
    from gojo import interfaces
    
    DEVICE = 'cuda'


.. parsed-literal::

    C:\Users\fgarcia\anaconda3\envs\mlv0\lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


Data loading
------------

.. code:: python

    # FashionMNIST labels (https://github.com/zalandoresearch/fashion-mnist)
    labels = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }
    
    # define the transformation used to load the images
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])
    
    # download the FashionMNIST datasets (train/test)
    train_dataset = datasets.FashionMNIST(
        root=os.path.join(os.path.expanduser('~'), 'test_datasets', 'pytorch'), 
        train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(
        root=os.path.join(os.path.expanduser('~'), 'test_datasets', 'pytorch'), 
        train=False, transform=transform, download=True)

.. code:: python

    # plot some examples
    np.random.seed(1997)
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    for i, ax in enumerate(axes.flatten()):
        img, label = train_dataset[np.random.choice(len(train_dataset))]
        img_np = img.numpy().squeeze(0)    
        
        ax.imshow(img_np, cmap='Greys')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(labels[label])
    plt.show()



.. image:: Example_4_Vanilla_Variational_Autoencoder_files/Example_4_Vanilla_Variational_Autoencoder_4_0.png


.. code:: python

    # separate the training data into train (85%) and validation (15%)
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, 
        [
            int(len(train_dataset) * 0.85),
            int(len(train_dataset) * 0.15)
        ],
        torch.Generator().manual_seed(1997)
    )
    
    print('Train: %d' % len(train_dataset))
    print('Valid: %d' % len(valid_dataset))
    print('Test: %d' % len(test_dataset))
    
    # create the dataloaders
    train_dl = DataLoader(
        train_dataset, 
        batch_size=1028, shuffle=True)
    
    valid_dl = DataLoader(
        valid_dataset, 
        batch_size=4096, shuffle=False)
    
    test_dl = DataLoader(
        test_dataset, 
        batch_size=4096, shuffle=False)


.. parsed-literal::

    Train: 51000
    Valid: 9000
    Test: 10000


Model training
--------------

.. code:: python

    # create a simple variational autoencoder
    model = deepl.models.VanillaVAE(
        encoder=torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
        
            torch.nn.Conv2d(16, 32, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(32, 16, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.MaxPool2d(2),
        
            torch.nn.Conv2d(16, 8, (2, 2)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
        
            torch.nn.Flatten()
        ),
        encoder_out_dim=128,
        decoder=torch.nn.Sequential(
            torch.nn.Linear(128, 288),
            torch.nn.ReLU(),
    
            torch.nn.Unflatten(1, (8, 6, 6)),
    
            torch.nn.ConvTranspose2d(8, 16, kernel_size=(4, 4)),
            torch.nn.ReLU(),
    
            torch.nn.ConvTranspose2d(16, 32, kernel_size=(4, 4)),
            torch.nn.ReLU(),
    
            torch.nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=2),
            torch.nn.ReLU(),
    
            torch.nn.ConvTranspose2d(16, 8, kernel_size=(2, 2), stride=1),
            torch.nn.ReLU(),
    
            torch.nn.ConvTranspose2d(8, 1, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU(),
        ),
        decoder_in_dim=128,
        latent_dim=10
    )
    
    # with torch.no_grad():
    #     tX = torch.Tensor(X[:12])
    #     out = model(tX)
    # out[0].shape, out[1]['mu'].shape, out[1]['logvar'].shape

.. code:: python

    output = deepl.fitNeuralNetwork(
        deepl.iterUnsupervisedEpoch,
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        n_epochs=40,
        loss_fn=deepl.loss.ELBO(kld_weight=0.00005),
        optimizer_class=torch.optim.Adam,
        optimizer_params={'lr': 0.0001},
        device=DEVICE,
    )


.. parsed-literal::

    Training model...: 100%|███████████████████████████████████████████████████████████████| 40/40 [02:30<00:00,  3.77s/it]


Model convergence
~~~~~~~~~~~~~~~~~

.. code:: python

    train_info = output['train']
    valid_info = output['valid']

.. code:: python

    plotting.linePlot(
        train_info, valid_info,
        x='epoch', y='loss (mean)', err='loss (std)',
        labels=['Train', 'Validation'],
        title='Model convergence',
        ls=['solid', 'dashed'],
        style='default', legend_pos='center right'
    )



.. image:: Example_4_Vanilla_Variational_Autoencoder_files/Example_4_Vanilla_Variational_Autoencoder_11_0.png


.. code:: python

    plotting.linePlot(
        train_info, valid_info,
        x='epoch', y='reconstruction_loss',
        labels=['Train', 'Validation'],
        title='Model convergence (reconstruction loss)',
        ls=['solid', 'dashed'],
        style='default', legend_pos='center right'
    )



.. image:: Example_4_Vanilla_Variational_Autoencoder_files/Example_4_Vanilla_Variational_Autoencoder_12_0.png


.. code:: python

    plotting.linePlot(
        train_info, valid_info,
        x='epoch', y='KLD',
        labels=['Train', 'Validation'],
        title='Model convergence (KL-divergence)',
        ls=['solid', 'dashed'],
        style='default', legend_pos='center right'
    )



.. image:: Example_4_Vanilla_Variational_Autoencoder_files/Example_4_Vanilla_Variational_Autoencoder_13_0.png


Model evaluation
----------------

Analyze the reconstruction error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # load some examples
    np.random.seed(1000)
    model = model.eval()
    
    n_examples = 5
    
    tX_test_ex = []
    label_test_ex = []
    test_indices = []
    
    tX_train_ex = []
    label_train_ex = []
    train_indices = []
    for i in range(n_examples):
        # get test image
        idx = np.random.choice(len(test_dataset))
        test_indices.append(idx)
        img, label = test_dataset[idx]
        tX_test_ex.append(img)
        label_test_ex.append(labels[label])
        
        # get train image
        idx = np.random.choice(len(train_dataset))
        train_indices.append(idx)
        img, label = train_dataset[idx]
        tX_train_ex.append(img)
        label_train_ex.append(labels[label])
        
    tX_test_ex = torch.cat(tX_test_ex)
    tX_train_ex = torch.cat(tX_train_ex)
    
    # reconstruct input
    with torch.no_grad():
        r_tX_test_ex = model(tX_test_ex.unsqueeze(1).to(device=DEVICE))[0]
        r_tX_train_ex = model(tX_train_ex.unsqueeze(1).to(device=DEVICE))[0]
        
    r_X_test_ex = r_tX_test_ex.squeeze(1).cpu().numpy()
    r_X_train_ex = r_tX_train_ex.squeeze(1).cpu().numpy()

.. code:: python

    # display exapmles for the training data
    fig, axes = plt.subplots(n_examples, 2, figsize=(8, 3*n_examples))
    fig.tight_layout()
    for i in range(len(axes)):
        for ii in range(len(axes[i])):
            ax = axes[i, ii]
            if ii == 0:
                ax.imshow(r_X_train_ex[i], cmap='Greys')
                ax.set_title('Reconstructed (train)')
            else:
                ax.imshow(train_dataset[train_indices[i]][0].numpy().squeeze(), cmap='Greys')
                ax.set_title('Original (train)')



.. image:: Example_4_Vanilla_Variational_Autoencoder_files/Example_4_Vanilla_Variational_Autoencoder_17_0.png


.. code:: python

    # display exapmles for the test data
    fig, axes = plt.subplots(n_examples, 2, figsize=(8, 3*n_examples))
    fig.tight_layout()
    for i in range(len(axes)):
        for ii in range(len(axes[i])):
            ax = axes[i, ii]
            if ii == 0:
                ax.imshow(r_X_test_ex[i], cmap='Greys')
                ax.set_title('Reconstructed (test)')
            else:
                ax.imshow(test_dataset[test_indices[i]][0].numpy().squeeze(), cmap='Greys')
                ax.set_title('Original (test)')



.. image:: Example_4_Vanilla_Variational_Autoencoder_files/Example_4_Vanilla_Variational_Autoencoder_18_0.png


Compute regression metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    metric_stats = []
    for tX, tY in tqdm(test_dl):
        with torch.no_grad():
            rtX = model(tX.to(device=DEVICE))[0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metric_stats.append(core.getScores(
                tX.cpu().numpy().reshape(tX.shape[0], -1),
                rtX.cpu().numpy().reshape(tX.shape[0], -1),
                core.getDefaultMetrics('regression')
            ))
            
    metric_stats_df = pd.DataFrame(metric_stats)
    pd.DataFrame(metric_stats_df.mean(), columns=['Score']).round(decimals=3)


.. parsed-literal::

    100%|████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.30it/s]




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>explained_variance</th>
          <td>-10.559</td>
        </tr>
        <tr>
          <th>mse</th>
          <td>0.022</td>
        </tr>
        <tr>
          <th>mae</th>
          <td>0.081</td>
        </tr>
        <tr>
          <th>r2</th>
          <td>-122.206</td>
        </tr>
        <tr>
          <th>pearson_correlation</th>
          <td>0.327</td>
        </tr>
      </tbody>
    </table>
    </div>



Visualize embedding dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # generate model embeddings (using the mean vector)
    embeddings = []
    label_idx = []
    n_samples = 12
    samples = model.sample(n_samples, current_device=DEVICE)
    samples = samples.cpu().numpy().squeeze()
    for tX, label in tqdm(test_dl):
        with torch.no_grad():
            vae_emb = model.encode(tX.to(device=DEVICE))[0].cpu().numpy()
            
        embeddings.append(vae_emb)
        label_idx.append(label.numpy())
        
    embeddings = np.concatenate(embeddings)
    n_samples = 12
    samples = model.sample(n_samples, current_device=DEVICE)
    samples = samples.cpu().numpy().squeeze()
    label_idx = np.concatenate(label_idx)


.. parsed-literal::

    100%|████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  6.49it/s]


.. code:: python

    # generate 2d embeddings
    emb2d = umap.UMAP(n_components=2).fit_transform(embeddings)

.. code:: python

    fig, ax = plt.subplots()
    
    for i, label in labels.items():
        label_mask = label_idx == i
        ax.scatter(
            emb2d[label_mask, 0],
            emb2d[label_mask, 1],
            label=label, s=20
        )
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Embedding projection')
    ax.set_xlabel('Emb 0')
    ax.set_ylabel('Emb 1')
    ax.grid(alpha=0.2, color='grey')
    ax.legend(loc=(1.04, 0))
    plt.show()



.. image:: Example_4_Vanilla_Variational_Autoencoder_files/Example_4_Vanilla_Variational_Autoencoder_24_0.png


Generate samples
~~~~~~~~~~~~~~~~

.. code:: python

    n_samples = 12
    samples = model.sample(n_samples, current_device=DEVICE)
    samples = samples.cpu().numpy().squeeze()
    
    fig, axes = plt.subplots(n_samples // 3, 3, figsize=(10, 3 * (n_samples // 3)))
    fig.tight_layout()
    
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i], cmap='Greys')
        ax.set_xticks([])
        ax.set_yticks([])



.. image:: Example_4_Vanilla_Variational_Autoencoder_files/Example_4_Vanilla_Variational_Autoencoder_26_0.png

