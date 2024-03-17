'''
This module holds the primary functionality for the experiment manager

The Experiment manager is responsible for managing a directory where the results of experiments
can be recorded. It is designed to be used in a machine learning pipeline, where the results of
experiments are saved to a directory, and the code is backed up to a git repository.

Example experiment initialization:

    experiment = Experiment(
        experiments_root_dir, experiment_name, current_group_name, custom_id_generator, auto_commit_git, outputs_name,
        logging_module)
    experiment.initialize_experiment()
    torch.save(model.state_dict(), experiment.outputs_path / 'model_{epoch}.pth')

Example Experiment.record_info:

    dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    input_data_index = 0
    label_data_index = 1

    input_data_shape_str, label_data_shape_str = [", ".join(dataloader[data_index][0].shape[1:]) for data_index in [input_data_index, label_data_index]]

    my_experiment.record_info(f'input_shape: [batch_size, {input_data_shape_str}]')
    my_experiment.record_info(f'label_shape: [batch_size, {label_data_shape_str}]')

    This will be saved to experiments_root_dir[/experiment_group]/experiment_id/info.txt

'''

import logging
import pathlib
import typing
import os
import shutil
import datetime
import subprocess

LOGS_NAME = 'logs'


class Experiment:

    def __init__(self,
                 experiments_root_dir: str | os.PathLike,
                 experiment_name: str | os.PathLike,
                 current_group_name: str | os.PathLike,
                 auto_commit_git: bool = False,
                 outputs_name: str | os.PathLike = 'checkpoints',
                 logging_module: logging.Logger = 'default',
                 custom_id_generator: typing.Callable[[], str] = None,):

        self._experiments_root_dir = pathlib.Path(experiments_root_dir)
        self._outputs_name = outputs_name

        self._group_dir = self._experiments_root_dir / current_group_name

        self._experiment_name = experiment_name
        self._output_dir = None

        self._id_generator = custom_id_generator if custom_id_generator is not None else self._default_id_generator
        self._auto_commit_git = auto_commit_git

        if logging_module == 'default':
            self._log = logging.getLogger()
            self._log.setLevel(logging.DEBUG)  # Minimum level for the root logger
            # Create a file handler
            file_handler = logging.FileHandler()
            file_handler.setLevel(logging.INFO)  # File handler level
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self._log.addHandler(file_handler)
        elif logging_module is not None:
            self._log = logging_module
        else:
            self._log = logging.getLogger()
            self._log.setLevel(logging.DEBUG)

    @property
    def outputs_path(self):
        '''This property supplies the directory for the outputs of the experiment. (checkpoints, etc)'''
        return self._output_dir

    @property
    def logs_path(self):
        '''This property supplies the directory for the logs of the experiment.'''
        return self._logs_dir

    @property
    def experiment_root(self):
        '''This property supplies the root directory for the experiment, wherein groups
        and individual experiments are stored.'''
        return self._experiments_root_dir

    def initialize_experiment(self):
        '''recommended to run your pipeline with a sanity check using minimal data count to validate that everything
        is working, then start the experiment to back-up code state and initialize the experiment directories.
        1. build directories
        2. create new commit
        3. initialize log file in the directory with commit hash
        '''

        self._log.info(f'Initializing experiment: {self._group_dir.name}/{self._experiment_name}')

        self._experiment_directory = self._group_dir / self._id_generator()
        self._logs_dir = self._experiment_directory / LOGS_NAME
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = self._experiment_directory / self._outputs_name

        if self._auto_commit_git:
            self._log.info('auto-creating experiment commit...')
            self._do_auto_commit(f'Experiment auto-commit: {self._group_dir.name}/{self._experiment_name}')

    def _get_experiment_index(self):
        '''Virtual/Protected; decides how to name the new experiment'''
        experiment_count = 0
        if self._group_dir.exists():
            experiment_count = len(list(self._group_dir.iterdir()))
        return experiment_count

    def record_info(self, info:str) -> None:
        with open(self._experiment_directory / 'info.txt', 'a') as f:
            f.write(info + '\n')

    def _default_id_generator(self) -> str:
        experiment_index = self._get_experiment_index()
        return f'{experiment_index}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    def _record_commit_hash(self, commit_hash: str):
        self.record_info('commit_hash: ' + commit_hash)
    
    def _do_auto_commit(self, commit_message: str, push: bool = False):
        try:
            cur_cwd = os.getcwd()
            # Check if there are any changes to commit
            subprocess.run(["git", "add", "."], check=True, cwd=cur_cwd)
            subprocess.run(["git", "diff", "--staged", "--quiet"], check=False, cwd=cur_cwd)
            
            # Commit changes
            subprocess.run(["git", "commit", "-m", commit_message], check=True, cwd=cur_cwd)

            # Extract the commit hash of the most recent commit
            commit_hash_result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, cwd=cur_cwd, stdout=subprocess.PIPE, text=True)
            commit_hash = commit_hash_result.stdout.strip()

            # Log the commit hash to the experiment's directory
            self._record_commit_hash(commit_hash)

            # if push:
            #     # Push commit to remote repository
            #     subprocess.run(["git", "push"], check=True, cwd=cur_cwd)

        except subprocess.CalledProcessError as _e:
            self._log.exception('Git command failed: ')
