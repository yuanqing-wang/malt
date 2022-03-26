# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def get_default_trainer(
    optimizer: str="Adam",
    learning_rate: float=1e-3,
    n_epochs: int=10,
    batch_size: int=-1,
    validation_split: float=0.1,
    reduce_factor: float=0.5,
    warmup: int=50,
    without_player: bool=False,
    min_learning_rate: float=1e-6,
):
    """ Get the default training scheme for models.

    Parameters
    ----------
    optimizer : str
        Name of the optimizer. Must be an attribute of `torch.optim`

    learning_rate : float
        Initial learning rate.

    n_epochs : int
        Maximum epochs.

    batch_size : int
        Batch size.

    validation_split : float
        Proportion of validation set.

    reduce_factor : float
        Rate of learning rate reduction.

    Returns
    -------
    Callable : Trainer function.

    """
    def _default_trainer_without_player(
        model,
        ds,
        optimizer=optimizer,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        min_learning_rate=min_learning_rate,
        warmup=warmup,
    ):
        # see if cuda is available
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        # get original device
        original_device = next(model.parameters()).device

        # move model to cuda if available
        model = model.to(device)

        if len(ds) > warmup:
            # split into training and validation
            ds_tr, ds_vl = ds.split([1.0-validation_split, validation_split])

        else:
            ds_tr = ds
            ds_vl = ds

        # consider the case of one batch
        if batch_size == -1:
            batch_size = len(ds)

        # put data into loader
        ds_tr = ds_tr.view(batch_size=batch_size)
        ds_vl = ds_vl.view(batch_size=len(ds_vl))

        # get optimizer object
        optimizer = getattr(torch.optim, optimizer,)(
            model.parameters(),
            learning_rate,
        )

        # get scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=reduce_factor,
            patience=10,
        )

        # train
        model.train()
        for idx_epoch in range(n_epochs):  # loop through the epochs
            for x in ds_tr:  # loop through the dataset
                x = [_x.to(device) for _x in x]
                optimizer.zero_grad()
                loss = model.loss(*x).mean()  # average just in case
                loss.backward()
                optimizer.step()


            with torch.no_grad():
                x = next(iter(ds_vl))
                x = [_x.to(device) for _x in x]
                loss = model.loss(*x).mean()

                if idx_epoch > warmup:
                    scheduler.step(loss)
                    if optimizer.param_groups[0]['lr'] < min_learning_rate:
                        break

        x = next(iter(ds_tr))
        x = [_x.to(device) for _x in x]
        loss = model.loss(*x).mean()
        model = model.to(original_device)
        return model

    def _default_trainer(
        player,
        *args, **kwargs
    ):
        return _default_trainer_without_player(
            player.model,
            player.portfolio,
            *args, **kwargs,
        )

    if without_player is True:
        return _default_trainer_without_player
    else:
        return _default_trainer
