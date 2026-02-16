import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from src.utils.utils import instantiate_callbacks, instantiate_loggers

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Set Seed
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # 2. Instantiate DataModule
    print(f"Instantiating DataModule <{cfg.data._target_}>")
    datamodule = instantiate(cfg.data)

    # 3. Instantiate Model
    print(f"Instantiating Model <{cfg.model._target_}>")
    model = instantiate(cfg.model)

    # 4. Instantiate Callbacks & Loggers
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    loggers = instantiate_loggers(cfg.get("logger"))

    # 5. Instantiate Trainer
    print(f"Instantiating Trainer <{cfg.trainer._target_}>")
    trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # 6. Train the model
    print("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)

    # 7. Test the model
    print("Starting testing...")
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    main()