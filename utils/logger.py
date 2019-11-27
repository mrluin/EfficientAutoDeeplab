'''
@author: Jingbo Lin
@contact: ljbxd180612@gmail.com
@github: github.com/mrluin
'''
from pathlib import Path
import importlib, warnings
import os, sys, time, numpy as np
from io import BytesIO as BIO



class Logger(object):
    def __init__(self, log_dir, seed, create_model_dir=True,
                 use_tf=False):
        self.seed = int(seed)
        self.log_dir = Path(log_dir)
        self.model_dir = Path(log_dir) / 'checkpoint'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if create_model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)

        self.use_tf

        self.logger_path =
        self.logger_file =

        if self.use_tf:

        else:

    def __repr__(self):

    def path(self, mode):

    def extract_log(self):


    def close(self):


    def log(self, string, save=True, stdout=False):
        if stdout:
            sys.stdout.write(string); sys.stdout.flush()
        else:
            print(string)
        if save:

    def scalar_summary(self, tags, values, step):

