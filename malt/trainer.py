# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def get_default_trainer(
        optimizer="Adam",
        learning_rate=1e-3,
        n_epochs=10,
        batch_size=-1
    ):

    def _default_trainer(
            player,
            optimizer=optimizer,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
        ):
        # put data into loader
        ds = player.history.view(batch_size=batch_size)

        # get optimizer object
        optimizer = getattr(
            torch.optim
        )(
            player.net.parameters(),
            learning_rate,
        )

        # train
        for _ in range(n_epochs): # loop through the epochs
            for x in ds: # loop through the dataset
                optimizer.zero_grad()
                loss = player.net.loss(*x).mean() # average just in case
                loss.backward()
                optimizer.step()

        return player.net

    return _default_trainer
