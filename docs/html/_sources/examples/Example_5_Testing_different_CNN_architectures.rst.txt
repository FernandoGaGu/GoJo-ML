Example 6. Testing different CNN architectures
==============================================

.. code:: python

    import os
    import torch
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from tqdm import tqdm
    
    # GOJO libraries
    from gojo import deepl
    from gojo import plotting
    from gojo import core
    from gojo import util
    
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



.. image:: Example_5_Testing_different_CNN_architectures_files/Example_5_Testing_different_CNN_architectures_4_0.png


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
        batch_size=2048, shuffle=True)
    
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


Vanilla CNN
-----------

.. code:: python

    v_cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
        
            torch.nn.Conv2d(8, 16, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(16, 32, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2),
        
            torch.nn.Conv2d(32, 64, (2, 2)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.AvgPool2d(2),
            
            torch.nn.Flatten(),
        
            torch.nn.Linear(256, 64),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
    )
    
    print('Number of model trainable parameters: %d' % util.tools.getNumModelParams(v_cnn))


.. parsed-literal::

    Number of model trainable parameters: 31482


.. code:: python

    output_v_cnn = deepl.fitNeuralNetwork(
        deepl.iterSupervisedEpoch,
        model=v_cnn,
        train_dl=train_dl,
        valid_dl=valid_dl,
        n_epochs=40,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.Adam,
        optimizer_params={'lr': 0.0001},
        device=DEVICE,
    )


.. parsed-literal::

    Training model...: 100%|███████████████████████████████████████████████████████████████| 40/40 [02:06<00:00,  3.17s/it]


Model convergence
~~~~~~~~~~~~~~~~~

.. code:: python

    plotting.linePlot(
        output_v_cnn['train'], 
        output_v_cnn['valid'],
        x='epoch', y='loss (mean)', err='loss (std)',
        labels=['Train', 'Validation'],
        title='Model convergence',
        ls=['solid', 'dashed'],
        style='default', legend_pos='center right'
    )



.. image:: Example_5_Testing_different_CNN_architectures_files/Example_5_Testing_different_CNN_architectures_10_0.png


Model evaluation
~~~~~~~~~~~~~~~~

.. code:: python

    # make model predictions
    v_cnn = v_cnn.eval()
    y_hat = []
    y_true = []
    with torch.no_grad():
        for X, y in tqdm(test_dl, desc='Performing predictions...'):
            # convert logits to probabilities
            _y_hat = torch.nn.functional.softmax(v_cnn(X.to(device=DEVICE)), dim=1).cpu().numpy()
            
            # convert from one-hot encoding to integer-coding
            y_hat.append(_y_hat.argmax(axis=1))
            y_true.append(y.numpy())
            
    y_hat = np.concatenate(y_hat)
    y_true = np.concatenate(y_true)
    
    # calculate per-class accuracy
    acc_per_label = {}
    for label_key, label in labels.items():
        y_true_bin = (y_true == label_key).astype(int)
        y_hat_bin = (y_hat == label_key).astype(int)
        
        # compute accuracy
        acc_per_label[label] = core.getScores(
            y_true_bin,
            y_hat_bin,
            metrics=core.getDefaultMetrics('binary_classification', ['accuracy'])
        )['accuracy']
    
    print('Average accuracy: %.3f' % np.mean(list(acc_per_label.values())))
    pd.DataFrame([acc_per_label], index=['Accuracy']).T


.. parsed-literal::

    Performing predictions...: 100%|█████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  8.67it/s]

.. parsed-literal::

    Average accuracy: 0.975


.. parsed-literal::

    




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
          <th>Accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>T-shirt/top</th>
          <td>0.9656</td>
        </tr>
        <tr>
          <th>Trouser</th>
          <td>0.9951</td>
        </tr>
        <tr>
          <th>Pullover</th>
          <td>0.9651</td>
        </tr>
        <tr>
          <th>Dress</th>
          <td>0.9748</td>
        </tr>
        <tr>
          <th>Coat</th>
          <td>0.9588</td>
        </tr>
        <tr>
          <th>Sandal</th>
          <td>0.9922</td>
        </tr>
        <tr>
          <th>Shirt</th>
          <td>0.9303</td>
        </tr>
        <tr>
          <th>Sneaker</th>
          <td>0.9872</td>
        </tr>
        <tr>
          <th>Bag</th>
          <td>0.9908</td>
        </tr>
        <tr>
          <th>Ankle boot</th>
          <td>0.9893</td>
        </tr>
      </tbody>
    </table>
    </div>



CNN with residual connections
-----------------------------

.. code:: python

    res_cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, (3, 3), stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            
            deepl.cnn.ResNetBlock(8, 16, kernel_size=3),
            torch.nn.MaxPool2d(2),
        
            deepl.cnn.ResNetBlock(16, 32, kernel_size=3),
            torch.nn.MaxPool2d(2),
        
            torch.nn.Flatten(),
        
            torch.nn.Linear(288, 64),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
    )
    
    print('Number of model trainable parameters: %d' % util.tools.getNumModelParams(res_cnn))


.. parsed-literal::

    Number of model trainable parameters: 37546


.. code:: python

    output_res_cnn = deepl.fitNeuralNetwork(
        deepl.iterSupervisedEpoch,
        model=res_cnn,
        train_dl=train_dl,
        valid_dl=valid_dl,
        n_epochs=40,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.Adam,
        optimizer_params={'lr': 0.0001},
        device=DEVICE,
    )


.. parsed-literal::

    Training model...: 100%|███████████████████████████████████████████████████████████████| 40/40 [02:00<00:00,  3.00s/it]


Model convergence
~~~~~~~~~~~~~~~~~

.. code:: python

    plotting.linePlot(
        output_res_cnn['train'], 
        output_res_cnn['valid'],
        x='epoch', y='loss (mean)', err='loss (std)',
        labels=['Train', 'Validation'],
        title='Model convergence',
        ls=['solid', 'dashed'],
        style='default', legend_pos='center right'
    )



.. image:: Example_5_Testing_different_CNN_architectures_files/Example_5_Testing_different_CNN_architectures_17_0.png


Model evaluation
~~~~~~~~~~~~~~~~

.. code:: python

    # make model predictions
    res_cnn = res_cnn.eval()
    y_hat = []
    y_true = []
    with torch.no_grad():
        for X, y in tqdm(test_dl, desc='Performing predictions...'):
            # convert logits to probabilities
            _y_hat = torch.nn.functional.softmax(res_cnn(X.to(device=DEVICE)), dim=1).cpu().numpy()
            
            # convert from one-hot encoding to integer-coding
            y_hat.append(_y_hat.argmax(axis=1))
            y_true.append(y.numpy())
            
    y_hat = np.concatenate(y_hat)
    y_true = np.concatenate(y_true)
    
    # calculate per-class accuracy
    acc_per_label = {}
    for label_key, label in labels.items():
        y_true_bin = (y_true == label_key).astype(int)
        y_hat_bin = (y_hat == label_key).astype(int)
        
        # compute accuracy
        acc_per_label[label] = core.getScores(
            y_true_bin,
            y_hat_bin,
            metrics=core.getDefaultMetrics('binary_classification', ['accuracy'])
        )['accuracy']
    
    print('Average accuracy: %.3f' % np.mean(list(acc_per_label.values())))
    pd.DataFrame([acc_per_label], index=['Accuracy']).T


.. parsed-literal::

    Performing predictions...: 100%|█████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  8.13it/s]

.. parsed-literal::

    Average accuracy: 0.973


.. parsed-literal::

    




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
          <th>Accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>T-shirt/top</th>
          <td>0.9621</td>
        </tr>
        <tr>
          <th>Trouser</th>
          <td>0.9957</td>
        </tr>
        <tr>
          <th>Pullover</th>
          <td>0.9561</td>
        </tr>
        <tr>
          <th>Dress</th>
          <td>0.9771</td>
        </tr>
        <tr>
          <th>Coat</th>
          <td>0.9552</td>
        </tr>
        <tr>
          <th>Sandal</th>
          <td>0.9910</td>
        </tr>
        <tr>
          <th>Shirt</th>
          <td>0.9275</td>
        </tr>
        <tr>
          <th>Sneaker</th>
          <td>0.9868</td>
        </tr>
        <tr>
          <th>Bag</th>
          <td>0.9924</td>
        </tr>
        <tr>
          <th>Ankle boot</th>
          <td>0.9903</td>
        </tr>
      </tbody>
    </table>
    </div>


