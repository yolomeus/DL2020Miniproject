import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from lightning_wrapper import LightningModel


@hydra.main(config_path='conf/config.yaml')
def train(cfg: DictConfig):
    """Train a pytorch model specified by the config file"""

    seed_everything(cfg.random_seed)

    model = LightningModel(cfg)

    train_cfg = cfg.training
    checkpoint_callback = ModelCheckpoint(save_top_k=train_cfg.save_ckpts,
                                          monitor=train_cfg.monitor,
                                          mode=train_cfg.mode)

    trainer = Trainer(max_epochs=train_cfg.epochs,
                      gpus=cfg.gpus,
                      deterministic=True,
                      checkpoint_callback=checkpoint_callback)
    trainer.fit(model)

    # test best
    best = list(trainer.checkpoint_callback.best_k_models.keys())[-1]
    trainer.model.load_from_checkpoint(best)
    trainer.test()


if __name__ == '__main__':
    train()
