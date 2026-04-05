import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import StochasticWeightAveraging

from src.config import MODEL_CACHE, OUTPUT_PATH
from src.datasets import CommonLitDataModule
from src.models import CommonLitModel
from src.utils import prepare_args, prepare_loggers_and_callbacks, resume_helper

torch.hub.set_dir(MODEL_CACHE)


def run_fold(fold: int, args):
    pl.seed_everything(args.seed + fold)
    resume, run_id = resume_helper(args)

    monitor_list = [("rmse", "min", None)]
    # Set up logging (TensorBoard and/or Weights & Biases) and callbacks (early stopping, model checkpointing, etc.)
    loggers, callbacks = prepare_loggers_and_callbacks(
        args.timestamp,
        args.model_name,
        fold,
        monitors=monitor_list,
        tensorboard=args.logging,
        wandb=args.logging,
        patience=None,
        run_id=run_id,
        save_weights_only=True,
    )

    # Optionally add Stochastic Weight Averaging (SWA) callback for improved generalization
    if args.swa:
        swa = StochasticWeightAveraging(swa_epoch_start=0.5)
        callbacks.append(swa)

    # Create the model and trainer, then fit the model using the data module
    model = CommonLitModel(**args.__dict__)

    trainer_kwargs = dict(
        logger=loggers,
        callbacks=callbacks,
        resume_from_checkpoint=resume,
    )

    if hasattr(args, "accelerator"):
        trainer_kwargs["accelerator"] = args.accelerator
    if hasattr(args, "devices"):
        trainer_kwargs["devices"] = args.devices

    if "accelerator" not in trainer_kwargs:
        if torch.backends.mps.is_available():
            trainer_kwargs["accelerator"] = "mps"
            trainer_kwargs["devices"] = 1
        elif torch.cuda.is_available():
            trainer_kwargs["accelerator"] = "gpu"
            trainer_kwargs["devices"] = getattr(args, "gpus", 1)
        else:
            trainer_kwargs["accelerator"] = "cpu"
            trainer_kwargs["devices"] = 1

    trainer = pl.Trainer().from_argparse_args(args, **trainer_kwargs)

    # Load data via the data module, which handles tokenization and batching
    dm = CommonLitDataModule().from_argparse_args(args)
    dm.setup("fit", fold)

    # Save tokenizer
    folder = args.model_name
    if "/" in folder:
        folder = folder.replace("/", "_")

    save_path = OUTPUT_PATH / args.timestamp / folder / f"fold_{fold}"
    dm.tokenizer.save_pretrained(save_path)
    model.config.to_json_file(str(save_path / "config.json"))

    # trainer.tune(model, datamodule=dm)  # Use with auto_lr_find
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    args = prepare_args()
    run_fold(args.fold - 1, args)
