import logging
from datetime import datetime
import torch
import segmentation_models_pytorch as smp
from catalyst import dl
from catalyst.engines.torch import CPUEngine, GPUEngine

from src.config import config
from src.base_config import Config
from src.tools import set_global_seed
from src.dataset import get_loaders
from src.config import NUM_CLASSES
from src.loggers import ClearMLLogger


seg_losses = config.seg_losses
cls_losses = config.cls_losses
loss_seg = "loss_" + seg_losses["name"]
loss_cls = "loss_" + cls_losses["name"]

criterion = {
    seg_losses["name"]: seg_losses["loss_fn"],
    cls_losses["name"]: cls_losses["loss_fn"],
}

train_callbacks = [
    dl.CriterionCallback(
        input_key="logits",             
        target_key="mask",     
        metric_key=loss_seg,
        criterion_key=seg_losses["name"],
    ),
    dl.CriterionCallback(
        input_key="logits",                      
        target_key="mask",                      
        metric_key=loss_cls,                   
        criterion_key=cls_losses["name"]                 
    ),
    dl.MetricAggregationCallback(
        metric_key="loss",
        mode="weighted_sum",
        metrics={loss_seg: seg_losses["weight"], loss_cls: cls_losses["weight"]},
    ),
    dl.BatchTransformCallback(
        scope="on_batch_end",
        transform="torch.sigmoid",
        input_key="logits",
        output_key="pred_mask",
    ),
    dl.DiceCallback(
        input_key="pred_mask", 
        target_key="mask",
    ),
    dl.CheckpointCallback(
        logdir=config.checkpoints_dir,
        loader_key='valid',
        metric_key=config.valid_metric,
        minimize=config.minimize_metric,
    ),
    dl.EarlyStoppingCallback(
        patience=config.early_stop_patience,
        loader_key='valid',
        metric_key=config.valid_metric,
        minimize=config.minimize_metric,
    ),
]

test_callbacks = [
    dl.BatchTransformCallback(
        scope="on_batch_end",
        transform="torch.sigmoid",
        input_key="logits",
        output_key="pred_mask",
    ),
    dl.DiceCallback(
        input_key="pred_mask", 
        target_key="mask",
    ),
]


def train(config: Config, clearml: bool = True):
    loaders, infer_loader = get_loaders(config)

    model = smp.Unet(encoder_name=config.model_kwargs["encoder_name"],
                     encoder_weights=config.model_kwargs["encoder_weights"],
                     classes=NUM_CLASSES,
                     aux_params={'pooling': 'avg', 'dropout': 0.2, 'classes': NUM_CLASSES})

    optimizer = config.optimizer(params=model.parameters(), **config.optimizer_kwargs)
    scheduler = config.scheduler(optimizer=optimizer, **config.scheduler_kwargs)
    if clearml:
        clearml_logger = ClearMLLogger(config)
        loggers={"_clearml": clearml_logger}
    else:
        loggers = None
    
    if torch.cuda.is_available():
        engine = GPUEngine()
    else:
        engine = CPUEngine()



    runner = dl.SupervisedRunner(
        input_key="image", 
        target_key="mask", 
        output_key=("logits", "pred_labels"),
    )


    runner.train(
        model=model,
        engine=engine,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=train_callbacks,
        loggers=loggers,
        num_epochs=config.n_epochs,
        valid_loader='valid',
        valid_metric=loss_seg,
        minimize_valid_metric=config.minimize_metric,
        seed=config.seed,
        verbose=True,
        load_best_on_end=True,
    )

    metrics = runner.evaluate_loader(
        model=model,
        loader=infer_loader["infer"],
        callbacks=test_callbacks,
        verbose=True,
        seed=config.seed,
    )
    
    
    if clearml:
        clearml_logger.log_metrics(metrics, scope="loader", runner=runner, infer=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    set_global_seed(config.seed)
    train(config)

