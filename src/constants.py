from os import path as osp


DATA_PATH = "/storage/minin/datasets/barcodes/my_dataset"
DF_PATH = osp.join(DATA_PATH, "bbox.tsv")
TRAIN_IMAGES_PATH = osp.join(DATA_PATH, "Barcodes")
PROJECT_PATH = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), "../"))  # noqa: WPS221
TORCH_FILE = osp.join(PROJECT_PATH, "weights/model.best.pth")
ONNX_FILE = osp.join(PROJECT_PATH, "weights/model.best.onnx")
