"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# Standard Library
import contextlib
import logging
import time
from datetime import timedelta
from io import StringIO
from typing import Optional

logging.basicConfig()


class ElapsedFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = record.created - self.start_time
        elapsed = timedelta(seconds=elapsed_seconds)
        return "{} - {}".format(elapsed, record.getMessage())


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def set_logging_level(level):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(ElapsedFormatter())
    logger.addHandler(handler)

    if "level" == "debug":
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            if "megapose" in logger.name:
                logger.setLevel(logging.DEBUG)
    else:
        pass
    return
