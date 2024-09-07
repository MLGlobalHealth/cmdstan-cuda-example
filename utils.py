import re
import os
import logging
import numpy as np
import arviz as az
import scipy.stats as stats
from cmdstanpy import CmdStanModel
from pathlib import Path
from shutil import copy, move
from datetime import datetime

class stan_model:
    """A thin wrapper around CmdStanModel to compile, sample, diagnose, and save."""
    def __init__(self, stan_file):
        self.stan_file = stan_file
        self.model_name = stan_file.stem
        self.stan_dir = stan_file.parent
        self.log_file = self.stan_dir / f'{self.model_name}.log'
        self.logger, self.handler = self._setup_logger()

    def _setup_logger(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        
        logger = logging.getLogger("cmdstanpy")
        logger.handlers = []
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                "%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        return logger, handler

    def change_namespace(self, user_header, name):
        filepath = user_header
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        with open(filepath, "w", encoding='utf-8') as file:
            for line in lines:
                # Matches all characters except a single space character
                match = re.match(r'^namespace (\S+)_model_namespace {$', line)
                if match:
                    new_line = f'namespace {name}_model_namespace {{\n'
                    file.write(new_line)
                else:
                    file.write(line)

    def compile(self, user_header=None, stanc_options=None, cpp_options=None, **kwargs):
        bnb_hpp_dir = user_header.parent / 'bnb'
        for file in Path(bnb_hpp_dir).rglob('*.hpp'):
            self.change_namespace(file, self.model_name)

        # Compile the Stan model
        print(f"Start compiling the model {self.stan_file}")
        self.model = CmdStanModel(stan_file=self.stan_file, 
                                            force_compile=True, 
                                            user_header=user_header, 
                                            stanc_options=stanc_options,
                                            cpp_options=cpp_options,
                                            **kwargs)

    def sample(self, data, **kwargs):
        self.data = data
        self.fit = self.model.sample(data=data, **kwargs)
        self.df = self.fit.draws_pd()

        separator = "=" * 100
        self.logger.debug(separator)
        self.logger.debug("="*40+" Sampling completed "+"="*40)
        self.logger.debug(separator)
        for index, file_path in enumerate(self.fit.runset.stdout_files, start=1):
            with open(file_path, 'r', encoding='utf-8') as out_file:
                content = out_file.read()
                self.logger.debug('Runset Stdout Files (self.fit.runset.stdout_files[%d]): %s', index, content)

    def diagnose(self):
        self.diagnosis = self.fit.diagnose()
        print(self.diagnosis)
        
        self.az_data = az.from_cmdstanpy(
            posterior=self.fit,
            posterior_predictive=None,
            log_likelihood="log_lik",
            observed_data={"y": self.data["y"]},
            save_warmup=False,
        )
        self.loo = az.loo(self.az_data, pointwise=True, scale="log")
        self.waic = az.waic(self.az_data, pointwise=True, scale="log")
        print(self.loo)
        print(self.waic)

        self.logger.debug("DataFrame (self.df):\n%s", self.df)
        self.logger.debug("LOO (self.loo):\n%s", self.loo)
        self.logger.debug("WAIC (self.waic):\n%s", self.waic)
        self.logger.debug("Diagnosis (self.diagnosis):\n%s", self.diagnosis)

    def save(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()
        logging.shutdown()

        # Setup save directory and save Stan csv
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        stan_save_dir = os.path.join(self.stan_dir, f"{self.model_name}_{timestamp}")
        if not os.path.exists(stan_save_dir):
            os.makedirs(stan_save_dir)
        self.fit.save_csvfiles(dir=stan_save_dir)

        copy(self.stan_file, stan_save_dir)
        move(self.log_file, stan_save_dir)

        # Clean temporary files
        # exe_file = os.path.join(self.stan_dir, self.model_name)
        # if os.path.isfile(exe_file):
        #     os.remove(exe_file)


def beta_neg_binomial_rng(r, alpha, beta, y_max, size=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    p = stats.beta.rvs(alpha, beta, size=size)
    # N Number of successes, p probability of success
    y = stats.nbinom.rvs(n=r,p=p)
    y = np.minimum(y, y_max)
    return y
