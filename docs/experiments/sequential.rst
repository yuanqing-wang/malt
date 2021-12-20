Sequential acquisition demo
---------------------------

Here we demonstrate how to use MALT for a sequential acquisition
experiment.

Open in Colab: http://data.wangyq.net/malt_notebooks/sequential.ipynb

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
    ⏲ Done in 0:00:46
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

    Using backend: pytorch


Grab data
---------

We download the ESOL dataset
(https://pubs.acs.org/doi/10.1021/ci034243x), randomly shuffle it and
partition into training, validation, and test (80:10:10).

.. code:: python

    data = malt.data.collections.esol()


.. parsed-literal::

    Downloading /root/.dgl/ESOL.zip from https://data.dgl.ai/dataset/ESOL.zip...
    Extracting file to /root/.dgl/ESOL
    Processing dgl graphs from scratch...
    Processing molecule 1000/1128


Define model
------------

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


Define agents
-------------

.. code:: python

    merchant = malt.agents.merchant.DatasetMerchant(data)
    assayer = malt.agents.assayer.DatasetAssayer(data)

Define player
-------------

.. code:: python

    player = malt.agents.player.SequentialModelBasedPlayer(
        model=model,
        policy=malt.policy.Greedy(),
        trainer=malt.trainer.get_default_trainer(),
        merchant=merchant,
        assayer=assayer,
    )

Run the experiment
------------------

.. code:: python

    while True:
        if player.step() is None:
            break


.. parsed-literal::

    /content/malt/malt/models/regressor.py:158: UserWarning: torch.cholesky is deprecated in favor of torch.linalg.cholesky and will be removed in a future PyTorch release.
    L = torch.cholesky(A)
    should be replaced with
    L = torch.linalg.cholesky(A)
    and
    U = torch.cholesky(A, upper=True)
    should be replaced with
    U = torch.linalg.cholesky(A).transpose(-2, -1).conj().
    This transform will produce equivalent results for all valid (symmetric positive definite) inputs. (Triggered internally at  ../aten/src/ATen/native/BatchLinearAlgebra.cpp:1285.)
      l_low = torch.cholesky(k_plus_sigma)


.. code:: python

    import numpy as np
    y = np.array(player.portfolio.y)

.. code:: python

    from matplotlib import pyplot as plt
    plt.plot(y)
    plt.xlabel("steps")




.. parsed-literal::

    Text(0.5, 0, 'steps')




.. image:: sequential_files/sequential_16_1.png


.. code:: python

    plt.plot(np.maximum.accumulate(y))
    plt.xlabel("steps")
    plt.ylabel("$y_\mathtt{max}$")




.. parsed-literal::

    Text(0, 0.5, '$y_\\mathtt{max}\x10$')




.. image:: sequential_files/sequential_17_1.png


