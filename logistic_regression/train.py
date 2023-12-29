import logging
import os
from enum import Enum

import git
import hydra
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import skops.io as sio
from dvc.repo import Repo
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.tensorboard import SummaryWriter

TRAIN_FILE_PATH = "../data/train.csv"
TEST_FILE_PATH = "../data/test.csv"
MODEL_PATH = "../models/model.skops"


class Stage(Enum):
    TRAIN = "train"
    TEST = "test"


class TrainModel:
    def __init__(self, cfg: DictConfig, stage: Stage, model=None):
        self.cfg = cfg
        self.data = None
        self.stage = stage
        self.model = model
        self.input_file = TRAIN_FILE_PATH if stage == Stage.TRAIN else TEST_FILE_PATH
        self.X = None
        self.y = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self):
        """
        Инициализирует признаки X и target Y в зависимости от выбранного Stage

        """
        self.logger.info("Load data from DVC")

        # Загрузка данных из csv файла с помощью DVC и Google Drive

        self.data = pd.read_csv(self.input_file)

        # берем как признаки все столбцы кроме последнего

        self.X = self.data.iloc[:, :-1]

        # Для target последний столбец

        self.y = self.data.iloc[:, -1]

    def fit_data(self):
        """
        Метод для инициализации и обучения модели на стадии Stage.TRAIN

        """

        self.logger.info("Fit data and train model started")
        self.model = LogisticRegression(
            tol=self.cfg.model.tolerance,
            random_state=self.cfg.model.random_state,
            max_iter=self.cfg.model.max_iter,
        )

        # обучаем модель на данных
        self.model.fit(self.X, self.y)

        self.log_experiment()

        self.logger.info("Fit data and train model finished")

    def log_experiment(self):
        """
        Логирование эксперимента в mlflow, tensorboard

        """
        repo = git.Repo(search_parent_directories=True)
        git_commit_id = repo.head.object.hexsha

        # Set up MLflow

        mlflow.set_tracking_uri(self.cfg.mlflow.host)
        mlflow.set_experiment(self.cfg.mlflow.experiment)

        # Set up TensorBoard

        writer = SummaryWriter(log_dir=self.cfg.tensorboard.folder_path)

        with mlflow.start_run(
            run_name=self.cfg.mlflow.run_name + "_" + self.stage.value
        ):
            # логируем параметры модели
            mlflow.log_params(
                {
                    "random_state": self.cfg.model.random_state,
                    "max_iter": self.cfg.model.max_iter,
                    "tolerance": self.cfg.model.tolerance,
                    "git_commit_id": git_commit_id,
                }
            )
            # cчитаем метрики

            y_pred_hat = self.model.predict_proba(self.X)

            # оставляем положительные вероятности

            y_pred_hat = y_pred_hat[:, 1]
            fpr, tpr, _ = roc_curve(self.y, y_pred_hat)
            precision, recall, _ = precision_recall_curve(self.y, y_pred_hat)
            roc_auc = roc_auc_score(self.y, y_pred_hat)
            y_pred = self.model.predict(self.X)
            f1 = f1_score(self.y, y_pred)
            pr_auc = average_precision_score(self.y, y_pred_hat)
            conf_matrix = confusion_matrix(self.y, y_pred)

            # логируем метрики в логах, tensorboard и mlflow

            self.logger.info(f"Roc AUC {roc_auc}")
            self.logger.info(f"F1 {f1}")
            self.logger.info(f"PR AUC {pr_auc}")
            writer.add_scalar("Roc AUC", roc_auc)
            writer.add_scalar("F1", f1)
            writer.add_scalar("PR AUC", pr_auc)

            mlflow.log_metrics({"Roc AUC": roc_auc, "PR AUC": pr_auc, "F1": f1})

            # строим графики

            plt.figure(figsize=(10, 7))
            plt.imshow(conf_matrix, cmap="Blues")
            plt.colorbar()
            writer.add_figure("Confusion matrix", plt.gcf())

            plt.plot(recall, precision, marker="*")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("PR curve")
            writer.add_figure("PR curve", plt.gcf())

            plt.plot(fpr, tpr, marker=".")
            plt.title("ROC curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            writer.add_figure("ROC curve", plt.gcf())
            writer.close()

    def save_model(self):
        sio.dump(self.model, MODEL_PATH)

        # Инициализируем репозиторий DVC
        repo = Repo(".")

        # Добавляем модель в DVC
        repo.add(MODEL_PATH)

        # Коммитим изменения
        repo.scm.commit("Added LogisticRegression model")

        # Пушим изменения в удаленный репозиторий
        repo.push()
        repo.close()

    def save_predicts_to_csv(self):
        y_pred = self.model.predict(X=self.X)
        y_pred.to_csv("predicts_result_on_test.csv")


def check_dvc():
    # список файлов для проверки
    files = [TRAIN_FILE_PATH, TEST_FILE_PATH, MODEL_PATH]

    # путь к репозиторию DVC,текущая
    dvc_repo_path = "."

    # создаем экземпляр репозитория DVC
    repo = Repo(dvc_repo_path)

    # проверяем наличие каждого файла
    for file in files:
        if not os.path.exists(file):
            # если файл не существует, делаем pull из DVC
            repo.pull(targets=[file])


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    check_dvc()
    model = TrainModel(cfg, stage=Stage.TRAIN)
    model.load_data()
    model.fit_data()
    model.save_model()


if __name__ == "__main__":
    main()
