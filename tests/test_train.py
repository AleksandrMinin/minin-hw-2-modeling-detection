import os

from train import train


def test_train(config_test):
    train(config_test, clearml=False)

    checkpoints_dir = config_test.checkpoints_dir + '/'
    assert os.path.exists(checkpoints_dir)

    for file_name in os.listdir(checkpoints_dir):
        file = checkpoints_dir + file_name
        if os.path.isfile(file):
            os.remove(file)

    os.rmdir(checkpoints_dir)
    assert not os.path.exists(checkpoints_dir)
