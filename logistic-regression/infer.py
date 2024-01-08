import hydra
import skops.io as sio
from omegaconf import DictConfig
from train import Stage, TrainModel, check_dvc


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    check_dvc(list(cfg.files.values()))
    model_from_file = sio.load(cfg.files.model, trusted=True)
    model = TrainModel(cfg, model=model_from_file, stage=Stage.TEST)
    model.load_data()
    model.log_experiment()
    model.save_predicts_to_csv()


if __name__ == "__main__":
    main()
