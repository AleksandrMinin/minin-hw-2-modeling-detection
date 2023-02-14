from os import path as osp




### PROJECT_PATH = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../'))
### CONFIGS_PATH = osp.join(PROJECT_PATH, 'configs')
DATA_PATH = '/storage/minin/datasets/barcodes/my_dataset'# osp.join(PROJECT_PATH, 'data')
DF_PATH = osp.join(DATA_PATH, 'bbox.tsv')
TRAIN_IMAGES_PATH = osp.join(DATA_PATH, 'Barcodes')
### EXPERIMENTS_PATH = osp.join(PROJECT_PATH, 'experiments')