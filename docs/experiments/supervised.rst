Supervised learning demo
------------------------

Open in Colab: http://data.wangyq.net/malt_notebooks/supervised.ipynb

Here we demonstrate how to use MALT for a simple supervised learning
experiment.

Intallation and imports
-----------------------

.. code:: python

    # install conda
    ! pip install -q condacolab
    import condacolab
    condacolab.install()


.. parsed-literal::

    ⏬ Downloading https://github.com/jaimergp/miniforge/releases/latest/download/Mambaforge-colab-Linux-x86_64.sh...
    📦 Installing...
    📌 Adjusting configuration...
    🩹 Patching environment...
    ⏲ Done in 0:00:34
    🔁 Restarting kernel...


.. code:: python

    %%capture
    ! mamba install rdkit
    ! rm -rf malt
    ! git clone https://github.com/yuanqing-wang/malt.git
    ! pip install dgl-cu101 dgllife

.. code:: python

    import torch
    import sys
    sys.path.append("/content/malt")
    import malt


.. parsed-literal::

    DGL backend not selected or invalid.  Assuming PyTorch for now.


.. parsed-literal::

    Setting the default backend to "pytorch". You can change it in the ~/.dgl/config.json file or export the DGLBACKEND environment variable.  Valid options are: pytorch, mxnet, tensorflow (all lowercase)


.. parsed-literal::

    Using backend: pytorch


Grab data
---------

We download the ESOL dataset
(https://pubs.acs.org/doi/10.1021/ci034243x), randomly shuffle it and
partition into training, validation, and test (80:10:10).



.. code:: python

    data = malt.data.collections.esol()


.. parsed-literal::

    Processing dgl graphs from scratch...
    Processing molecule 1000/1128


.. code:: python

    ds_tr, ds_vl, ds_te = data.split([8, 1, 1])

Model definition
----------------

.. code:: python

    model = malt.models.supervised_model.GaussianProcessSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=128,
        ),
        regressor=malt.models.regressor.ExactGaussianProcessRegressor(
            in_features=128, out_features=2,
        ),
        likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
    )


Train the model
---------------

.. code:: python

    trainer = malt.trainer.get_default_trainer(without_player=True, n_epochs=1000)
    model = trainer(model, ds_tr)


.. parsed-literal::

    GaussianProcessSupervisedModel(
      (representation): DGLRepresentation(
        (embedding_in): Sequential(
          (0): Linear(in_features=74, out_features=128, bias=True)
          (1): ReLU()
        )
        (gn0): GraphConv(in=128, out=128, normalization=both, activation=None)
        (gn1): GraphConv(in=128, out=128, normalization=both, activation=None)
        (gn2): GraphConv(in=128, out=128, normalization=both, activation=None)
        (embedding_out): Sequential(
          (0): Linear(in_features=128, out_features=128, bias=True)
        )
        (ff): Sequential(
          (0): Linear(in_features=128, out_features=128, bias=True)
        )
        (activation): ReLU()
      )
      (regressor): ExactGaussianProcessRegressor(
        (kernel): RBF()
      )
      (likelihood): HeteroschedasticGaussianLikelihood()
    )


Characterization
----------------

.. code:: python

    r2 = malt.metrics.supervised_metrics.R2()(model, ds_te)
    print(r2)
    
    rmse = malt.metrics.supervised_metrics.RMSE()(model, ds_te)
    print(rmse)


.. parsed-literal::

    tensor(0.8875, grad_fn=<RsubBackward1>)
    tensor(2.7404, grad_fn=<SqrtBackward0>)


.. parsed-literal::

    /content/malt/malt/metrics/base_metrics.py:10: UserWarning: Using a target size (torch.Size([112])) that is different to the input size (torch.Size([112, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
      return torch.sqrt(torch.nn.functional.mse_loss(target, input))


.. code:: python

    g, y = next(iter(ds_te.view(batch_size=len(ds_te))))

.. code:: python

    y_hat = model.condition(g).mean.flatten().cpu().detach().numpy()
    y = y.cpu().numpy()

.. code:: python

    from matplotlib import pyplot as plt
    plt.scatter(y, y_hat)




.. parsed-literal::

    <matplotlib.collections.PathCollection at 0x7f6fe06da910>




.. image:: supervised_files/supervised_17_1.png


