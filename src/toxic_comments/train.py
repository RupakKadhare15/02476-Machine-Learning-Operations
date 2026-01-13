import os
import hydra
import omegaconf
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from dotenv import load_dotenv
from datetime import datetime

from toxic_comments.datamodule import ToxicCommentsDataModule
from toxic_comments.model import ToxicCommentsTransformer


@hydra.main(version_base=None, config_path='../../configs', config_name='training.yaml')
def main(cfg):
    """Train the model."""

    load_dotenv()  # Load environment variables from .env file
    wandb.login()

    # convert cfg to a dict
    cfg_dict =  omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=cfg.wandb.entity,
        # Set the wandb project where this run will be logged.
        project=cfg.wandb.project,
        # Track hyperparameters and run metadata.
        config=cfg_dict,
    )

    # Initialize datamodule
    datamodule = ToxicCommentsDataModule(
        model_name_or_path=cfg.model_name_or_path,
        data_dir='data',
        train_batch_size=cfg.batch_size,
        eval_batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        num_workers=cfg.num_workers,
    )

    # Initialize model
    model = ToxicCommentsTransformer(
        model_name_or_path=cfg.model_name_or_path,
        num_labels=2,
        learning_rate=cfg.learning_rate,
        adam_epsilon=cfg.adam_epsilon,
    )


    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=cfg.patience,
        verbose=True,
        mode='min'
    )

    now = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir.split("outputs/")[-1]
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='models/' + now,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    # Train with PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=cfg.epochs,
                        callbacks=[early_stopping_callback, checkpoint_callback],
                        logger=WandbLogger(project=cfg.wandb.project),
    )
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()
