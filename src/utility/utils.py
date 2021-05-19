# -*- coding: utf-8 -*-
from enum import Enum
import logging
import sys


class LogFormatConst(Enum):
    BASIC = '[%(asctime)s] - {%(lineno)d} - %(name)s - %(levelname)s - %(message)s'
    FORMAT_1 = '[%(asctime)s] - {%(pathname)s:%(lineno)d} - %(name)s - %(levelname)s - %(message)s'
    FORMAT_2 = '[%(asctime)s] - p%(process)s - {%(pathname)s:%(lineno)d} - %(name)s - %(levelname)s - %(message)s'


class Logger:
    @property
    def log_format(self):
        return self._log_format

    @log_format.setter
    def log_format(self, log_format):
        self._log_format = log_format

    def __init__(self, log_format=None):
        if log_format is None:
            log_format = LogFormatConst.FORMAT_2.value
        self._log_format = log_format

    def get_logger(self, name, level=logging.INFO):
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.setLevel(level)
        formatter = logging.Formatter(self.log_format)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        return logger
