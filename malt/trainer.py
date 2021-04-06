# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def get_default_trainer(
    optimizer="Adam", learning_rate=1e-3, n_epochs=10, batch_size=-1
):
    def _default_trainer(
        player,
        optimizer=optimizer,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
    ):
        # see if cuda is available
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        # get original device
        original_device = next(player.model.parameters()).device

        # move model to cuda if available
        player.model = player.model.to(device)

        # consider the case of one batch
        if batch_size == -1:
            batch_size = len(player.portfolio)

        # put data into loader
        ds = player.portfolio.view(batch_size=batch_size)

        # get optimizer object
        optimizer = getattr(torch.optim, optimizer,)(
            player.model.parameters(),
            learning_rate,
        )

        # train
        for _ in range(n_epochs):  # loop through the epochs
            for x in ds:  # loop through the dataset
                x = [_x.to(device) for _x in x]
                optimizer.zero_grad()
                loss = player.model.loss(*x).mean()  # average just in case
                loss.backward()
                optimizer.step()

        player.model = player.model.to(original_device)
        return player.model

    return _default_trainer
