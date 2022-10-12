# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def get_default_trainer(
    optimizer: str = "Adam",
    learning_rate: float = 1e-3,
    n_epochs: int = 10,
    batch_size: int = -1,
    validation_split: float = 0.1,
    reduce_factor: float = 0.5,
    patience: int = 10,
    without_player: bool = False,
    min_learning_rate: float = 1e-6,
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
        data_train,
        data_valid,
        optimizer=optimizer,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        min_learning_rate=min_learning_rate,
        reduce_factor=reduce_factor,
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

        # consider the case of one batch
        if batch_size == -1:
            batch_size = len(data_train)

        # put data into loader
        data_train = data_train.view(batch_size=batch_size, pin_memory=True)
        data_valid = data_valid.view(batch_size=len(data_valid), pin_memory=True)

        # get optimizer object
        optimizer = getattr(torch.optim, optimizer,)(
            model.parameters(),
            learning_rate,
        )

        # get scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=reduce_factor,
            patience=patience,
        )

        # train
        for idx_epoch in range(n_epochs):  # loop through the epochs
            for x in data_train:  # loop through the dataset
                x = [_x.to(device) for _x in x]
                optimizer.zero_grad()
                model.train()
                loss = model.loss(*x).mean()  # average just in case
                loss.backward()
                optimizer.step()
                model.eval()

            with torch.no_grad():
                x = next(iter(data_valid))
                x = [_x.to(device) for _x in x]
                loss = model.loss(*x).mean()
                scheduler.step(loss)
                if optimizer.param_groups[0]['lr'] < min_learning_rate:
                    break

        x = next(iter(data_train))
        x = [_x.to(device) for _x in x]
        loss = model.loss(*x).mean()
        model = model.to(original_device)
        model.eval()
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
