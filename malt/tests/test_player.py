import pytest

def acquire_random_on_linear_alkane_with_player():
    import malt
    data = malt.data.collections.linear_alkanes(10)
    merchant = malt.agents.merchant.DatasetMerchant(data)
    assayer = malt.agents.assayer.DatasetAssayer(data)

    player = malt.agents.player.SequentialRandomPlayer(
        merchant=merchant,
        assayer=assayer,
    )

    player.step()
