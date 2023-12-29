import dvc.api
import pandas as pd


def load_data_from_dvc(data_path, repo_path):
    with dvc.api.open(
            data_path,
            repo=repo_path
    ) as fd:
        df = pd.read_csv(fd)
    return df


# Использование функции для загрузки данных
data = load_data_from_dvc(
    data_path='data/Social_Network_Ads.csv',  # Путь к вашему файлу данных в репозитории
    repo_path='.'  # Путь к вашему репозиторию DVC
)
