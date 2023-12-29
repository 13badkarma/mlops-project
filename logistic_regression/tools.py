import subprocess
from loguru import logger


class DvcHelper:

    def __init__(self, cfg):
        self.remote_folder = cfg.gdrive_folder
        self.remote_uri = cfg.gdrive_uri
        pass
