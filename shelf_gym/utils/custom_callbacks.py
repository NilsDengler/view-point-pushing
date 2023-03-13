from tqdm.auto import tqdm
import os
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from collections import deque

class SavingCallback(BaseCallback):
    def __init__(self, log_dir, save_freq=100000, verbose=0):
        super(SavingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            print("Save intermediate model and replay buffer")
            self.model.save(os.path.join(self.log_dir, 'intermediate_saved_model'))
            self.model.save_replay_buffer(os.path.join(self.log_dir, "replay_buffer"))

        return True

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.queue_steps = 100
        self.entropy_history = deque(self.queue_steps * [0], self.queue_steps)
        self.repeat_history = deque(self.queue_steps * [0], self.queue_steps)
        self.timeout_history = deque(self.queue_steps * [0], self.queue_steps)
        self.colissions_history = deque(self.queue_steps * [0], self.queue_steps)
        self.contact_history = deque(self.queue_steps * [0], self.queue_steps)
        self.entropy_change_history = deque(self.queue_steps * [0], self.queue_steps)
        self.step_history = deque(self.queue_steps * [0], self.queue_steps)
        self.reward_history = deque(self.queue_steps * [0], self.queue_steps)
        self.success_history = deque(self.queue_steps * [0], self.queue_steps)
        self.n_episodes_ = 0

    def _on_step(self) -> bool:
        #print(self.locals())
        self.check_for_var_existences()
        if self.training_env.envs[0].env.env.termination_info:
            self.n_episodes_ += 1
            # add new entry to history
            self.entropy_history.appendleft(self.training_env.envs[0].env.env.entropy_history[0])
            self.entropy_change_history.appendleft(self.training_env.envs[0].env.env.entropy_change)
            self.repeat_history.appendleft(int(self.training_env.envs[0].env.env.max_repeat_reached))
            self.timeout_history.appendleft(int(self.training_env.envs[0].env.env.max_step_reached))
            self.colissions_history.appendleft(int(self.training_env.envs[0].env.env.tilt_check))
            self.contact_history.appendleft(int(self.training_env.envs[0].env.env.contact_check))
            self.step_history.appendleft(int(self.training_env.envs[0].env.env.local_steps))
            self.reward_history.appendleft(self.training_env.envs[0].env.env.kommulative_reward)
            self.reward_history.appendleft(int(self.training_env.envs[0].env.env.is_success))
            #log history
            self.logger.record('rollout/colissions', np.sum(self.colissions_history)/self.queue_steps)
            self.logger.record('rollout/contacts', np.sum(self.contact_history)/self.queue_steps)
            self.logger.record('rollout/time_out', np.sum(self.timeout_history)/self.queue_steps)
            self.logger.record('rollout/max_repeat', np.sum(self.repeat_history)/self.queue_steps)
            self.logger.record('rollout/entropy', np.sum(self.entropy_history)/self.queue_steps)
            self.logger.record('rollout/entropy_change', np.sum(self.entropy_change_history)/self.queue_steps)
            self.logger.record('rollout/steps', np.sum(self.step_history)/self.queue_steps)
            self.logger.record('rollout/kom_reward', np.sum(self.reward_history)/self.queue_steps)
            self.logger.record('rollout/succes_per_episode', np.sum(self.success_history)/self.queue_steps)
        return True


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

class HParamCallback(BaseCallback):
    def __init__(self):
        """
        Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
        """
        super().__init__()

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
