import logging
import os
import sys
import time

class Logger():
    def __init__(self,Lever = "INFO"):
        self.logger = logging.getLogger()
        self.Lever = Lever
        self.logger.setLevel(level=Lever)

    def _get_formatter(self):
        fmt = logging.Formatter(fmt="%(asctime)s %(message)s")
        return fmt
    
    def _get_console_handler(self):
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(self._get_formatter())
        console_handler.setLevel(level=self.Lever)
        return console_handler
    
    def _get_file_handler(self, config):
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(config.output_dir, 'trainer.log'))
        file_handler.setFormatter(self._get_formatter())
        file_handler.setLevel(level=self.Lever)
        return file_handler
    
    def get_logger(self,config):
        self.logger.addHandler(self._get_console_handler())
        self.logger.addHandler(self._get_file_handler(config))
        return self.logger
