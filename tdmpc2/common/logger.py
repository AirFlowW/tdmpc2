import dataclasses
from collections import deque
from hashlib import sha256
import json
import subprocess
from pathlib import Path
import yaml
import os
import datetime
import re

import numpy as np
import pandas as pd
from termcolor import colored

from common import TASK_SET
from environments.video import ObservationVideoRecorder
from environments.episode_images import make_episode_image_recorder


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("episode_reward", "R", "float"),
	("episode_success", "S", "float"),
	("elapsed_time", "T", "time"),
]

CAT_TO_COLOR = {
	"pretrain": "yellow",
	"train": "blue",
	"eval": "green",
}


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg):
	"""
	Pretty-printing of current run information.
	Logger calls this method at initialization.
	"""
	prefix, color, attrs = "  ", "green", ["bold"]

	def _limstr(s, maxlen=36):
		return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

	def _pprint(k, v):
		print(
			prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v)
		)

	observations  = ", ".join([str(v) for v in cfg.obs_shape.values()])
	kvs = [
		("task", cfg.task_title),
		("steps", f"{int(cfg.steps):,}"),
		("observations", observations),
		("actions", cfg.action_dim),
		("experiment", cfg.exp_name),
	]
	w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
	div = "-" * w
	print(div)
	for k, v in kvs:
		_pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
	return lst if return_list else "-".join(lst)


def cfg_to_tags(cfg):
	"""Keep W&B tags within 64 characters while retaining distinct long names."""
	tags = cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"]
	return [tag if len(tag) <= 64 else f'{tag[:55]}-{sha256(tag.encode()).hexdigest()[:8]}'
			for tag in tags]


class VideoRecorder(ObservationVideoRecorder):
	"""Utility class for logging evaluation videos."""

	def __init__(self, cfg, wandb, fps=15):
		self.cfg = cfg
		super().__init__(cfg.work_dir / 'eval_video', wandb, 10 if cfg.task == 'pusht' else fps)


class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(self, cfg):
		self._log_dir = make_dir(cfg.work_dir)
		self._model_dir = make_dir(self._log_dir / "models")
		self._save_csv = cfg.save_csv
		self._save_agent = cfg.save_agent
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._eval = []
		self._records = {}
		window = cfg.get("episode_reward_avg_window", 100)
		if type(window) is not int or window < 1:
			raise ValueError("episode_reward_avg_window must be a positive integer")
		self._episode_rewards = deque(maxlen=window)
		resolved = json.loads(json.dumps(dataclasses.asdict(cfg), default=str))
		with open(cfg.work_dir / 'config.yaml', 'w') as stream:
			yaml.safe_dump(resolved, stream, sort_keys=False)
		revisions = {}
		repo_root = Path(__file__).resolve().parents[4]
		for name, directory in [('root', repo_root), ('tdmpc2', repo_root / 'forks/tdmpc2'),
				('sensorimotor', repo_root / 'forks/sensorimotor-world-model')]:
			result = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'],
				capture_output=True, text=True)
			revisions[name] = result.stdout.strip()
		with open(cfg.work_dir / 'revisions.json', 'w') as stream:
			json.dump(revisions, stream, indent=2)
		print_run(cfg)
		self.project = cfg.get("wandb_project", "none")
		self.entity = cfg.get("wandb_entity", "none")
		if not cfg.enable_wandb or self.project == "none" or self.entity == "none":
			print(colored("Wandb disabled.", "blue", attrs=["bold"]))
			self._wandb = None
			self._video = VideoRecorder(cfg, None) if cfg.save_video else None
			self._episode_images = self._make_episode_images(cfg)
			return
		os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
		import wandb

		wandb.init(
			project=self.project,
			entity=self.entity,
			name=cfg.get("wandb_name") or str(cfg.seed),
			group=self._group,
			tags=cfg_to_tags(cfg),
			dir=os.environ.get("WANDB_DIR", self._log_dir),
			config=dataclasses.asdict(cfg),
		)
		print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
		self._wandb = wandb
		for category, axis in (("train", "step"), ("eval", "step"), ("pretrain", "iteration")):
			wandb.define_metric(f"{category}/{axis}")
			wandb.define_metric(f"{category}/*", step_metric=f"{category}/{axis}")
		wandb.define_metric("videos/*", step_metric="eval/step")
		self._video = (
			VideoRecorder(cfg, self._wandb)
			if self._wandb and cfg.save_video
			else None
		)
		self._episode_images = self._make_episode_images(cfg)

	def _make_episode_images(self, cfg):
		if cfg.task in ('dot-avoidance', 'dot-navigation', 'pusht') and (cfg.save_video or cfg.get('save_episode_images', True)):
			return make_episode_image_recorder(cfg.task, cfg.work_dir / 'eval_images', wandb=self._wandb,
				max_episodes=cfg.get('episode_image_count', 5))
		return None

	@property
	def episode_images(self):
		return self._episode_images

	@property
	def video(self):
		return self._video

	@property
	def model_dir(self):
		return self._model_dir

	def save_agent(self, agent=None, identifier='final'):
		if self._save_agent and agent:
			fp = self._model_dir / f'{str(identifier)}.pt'
			agent.save(fp)
			if self._wandb:
				artifact = self._wandb.Artifact(
					self._group + '-' + str(self._seed) + '-' + str(identifier),
					type='model',
				)
				artifact.add_file(fp)
				# Include the config needed for standalone evaluation of downloaded models.
				artifact.add_file(self._log_dir / 'config.yaml')
				artifact.add_file(self._log_dir / 'revisions.json')
				upload = self._wandb.log_artifact(artifact)
				if identifier == 'pretrained':
					# Make the offline model available before any online updates begin.
					upload.wait(timeout=600)
					print(f'Pretrained model uploaded to W&B: {upload.qualified_name}')

	def finish(self, agent=None):
		try:
			self.save_agent(agent)
		except Exception as e:
			print(colored(f"Failed to save model: {e}", "red"))
		if self._wandb:
			self._wandb.finish()

	def log_latent_figures(self, figures, step):
		"""Mirror selected locally saved plots using the evaluation metric axis."""
		if self._wandb:
			self._wandb.log({
				'eval/step': step,
				**{f'eval/latent/figures/{name}': self._wandb.Image(str(path))
					for name, path in figures.items()},
			})

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.01f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise f"invalid log format type: {ty}"

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if k in d:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		print("   ".join(pieces))

	def pprint_multitask(self, d, cfg):
		"""Pretty-print evaluation metrics for multi-task training."""
		print(colored(f'Evaluated agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
		dmcontrol_reward = []
		metaworld_reward = []
		metaworld_success = []
		for k, v in d.items():
			if '+' not in k:
				continue
			task = k.split('+')[1]
			if task in TASK_SET['mt30'] and k.startswith('episode_reward'): # DMControl
				dmcontrol_reward.append(v)
				print(colored(f'  {task:<22}\tR: {v:.01f}', 'yellow'))
			elif task in TASK_SET['mt80'] and task not in TASK_SET['mt30']: # Meta-World
				if k.startswith('episode_reward'):
					metaworld_reward.append(v)
				elif k.startswith('episode_success'):
					metaworld_success.append(v)
					print(colored(f'  {task:<22}\tS: {v:.02f}', 'yellow'))
		dmcontrol_reward = np.nanmean(dmcontrol_reward)
		d['episode_reward+avg_dmcontrol'] = dmcontrol_reward
		print(colored(f'  {"dmcontrol":<22}\tR: {dmcontrol_reward:.01f}', 'yellow', attrs=['bold']))
		if cfg.task == 'mt80':
			metaworld_reward = np.nanmean(metaworld_reward)
			metaworld_success = np.nanmean(metaworld_success)
			d['episode_reward+avg_metaworld'] = metaworld_reward
			d['episode_success+avg_metaworld'] = metaworld_success
			print(colored(f'  {"metaworld":<22}\tR: {metaworld_reward:.01f}', 'yellow', attrs=['bold']))
			print(colored(f'  {"metaworld":<22}\tS: {metaworld_success:.02f}', 'yellow', attrs=['bold']))

	def log(self, d, category="train", *, episode_rewards=None):
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		if category == "train" and "episode_reward" in d:
			rewards = [d["episode_reward"]] if episode_rewards is None else episode_rewards
			self._episode_rewards.extend(float(reward) for reward in rewards)
			d = dict(d, episode_reward_moving_avg=float(np.mean(self._episode_rewards)))
		if self._wandb:
			_d = dict()
			for k, v in d.items():
				_d[category + "/" + k] = v
			# W&B owns the monotonic history step; phases use their own metric axes.
			self._wandb.log(_d)
		local = {}
		for key, value in d.items():
			if hasattr(value, 'detach'):
				value = value.detach().cpu().item()
			elif isinstance(value, np.generic):
				value = value.item()
			local[key] = value
		with open(os.path.join(self._log_dir, f'{category}.jsonl'), 'a') as stream:
			stream.write(json.dumps(local) + '\n')
		if self._save_csv:
			self._records.setdefault(category, []).append(local)
			pd.DataFrame(self._records[category]).to_csv(
				os.path.join(self._log_dir, f'{category}.csv'), index=False)

		self._print(d, category)
