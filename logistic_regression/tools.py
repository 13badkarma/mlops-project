import logging

from loguru import logger

# думал в начале о такой реализации, но сделал в итоге через pull
# def load_data_from_dvc(data_path, repo_path):
#     with dvc.api.open(
#             data_path,
#             repo=repo_path
#     ) as fd:
#         df = pd.read_csv(fd)
#     return df
#
#
# def load_model_from_dvc(model_path, repo_path):
#     file = dvc.api.read(model_path, repo=repo_path)
#     loaded_model = sio.load(file, trusted=True)
#     return loaded_model


# это я думал заюзать loguru вместо стандартного логера, но все оказалось сложнее :)
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Получить соответствующий уровень Loguru для сообщения
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Перенаправить сообщение в Loguru
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())
