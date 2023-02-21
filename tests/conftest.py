import pytest

from src.config import config


@pytest.fixture()#scope='session')
def config_test():
    config.n_epochs = 3
    config.df_path = './tests/data_for_tests/bbox.tsv'
    config.train_images_path = './tests/data_for_tests/Barcodes'
    config.train_size = 0.5
    config.batch_size = 1
    return config