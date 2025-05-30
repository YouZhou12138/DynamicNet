#!/usr/bin/env python3
"""Test Script
"""
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

from arg_setting.parse_setting import Parameter_setting
from datasets.data_selection import DATASETS
from models.model_selection import MODELS
from task.trainer_selection import Trainer


def test_model(setting_dict: dict, checkpoint: str=None):
    # Data
    data_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Data"].items()
    ]
    data_parser = ArgumentParser()
    data_parser = DATASETS[setting_dict["Setting"]].add_data_specific_args(data_parser)
    data_params = data_parser.parse_args(data_args)
    dm = DATASETS[setting_dict["Setting"]](data_params)

    # Model
    model_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Model"].items()
    ]
    model_parser = ArgumentParser()
    model_parser = MODELS[setting_dict["Architecture"]].add_model_specific_args(
        model_parser
    )
    model_params = model_parser.parse_args(model_args)
    model = MODELS[setting_dict["Architecture"]](model_params)

    # Task
    task_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Task"].items()
    ]
    task_parser = ArgumentParser()
    task_parser = Trainer[setting_dict["Setting"]].add_task_specific_args(task_parser)
    task_params = task_parser.parse_args(task_args)

    # 通过类名加载 checkpoint
    if checkpoint != "None":
        map_location = "cpu"
        task = Trainer[setting_dict["Setting"]].load_from_checkpoint(
            checkpoint_path=checkpoint,
            model=model,
            hparams=task_params,
            context_length=setting_dict["Task"]["context_length"],
            target_length=setting_dict["Task"]["target_length"],
            map_location=map_location,
        )
    else:
        task = Trainer[setting_dict["Setting"]](model=model, hparams=task_params)

    # Trainer

    trainer_dict = setting_dict["Trainer"]
    trainer_dict["logger"] = False
    trainer = pl.Trainer(callbacks=TQDMProgressBar(refresh_rate=10), **trainer_dict)

    dm.setup("test")

    trainer.test(model=task, datamodule=dm, ckpt_path=None)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "setting",
        type=str,
        metavar="path/to/setting.yaml",
        help="yaml with all settings",
    )
    parser.add_argument(
        "checkpoint", type=str, metavar="path/to/checkpoint", help="checkpoint file"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="moving_mnist",
        help="The code needs to know the training task based on the datasets",
    )

    parser.add_argument(
        "--pred_dir",
        type=str,
        default="/mnt/sda6/zhoulei1/ZY/Only_pf/experiments/en21x/DynamicNet7M/DynamicNet6M/config_seed=27/pred",
        metavar="path/to/prediction/dir",
        help="Path where to save predictions",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/autodl-tmp/ZY/dataset",
        metavar="path/to/dataset",
        help="Path where dataset is located",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        metavar="n gpus",
        default=1,
        help="how many gpus to use",
    )
    args = parser.parse_args()

    import os

    for k, v in os.environ.items():
        if k.startswith("SLURM"):
            del os.environ[k]

    setting_dict = Parameter_setting[args.data_name](args.setting)

    if args.pred_dir is not None:
        setting_dict["Task"]["pred_dir"] = args.pred_dir

    if args.data_dir is not None:
        setting_dict["Data"]["base_dir"] = args.data_dir

    if "gpus" in setting_dict["Trainer"]:
        setting_dict["Trainer"]["gpus"] = args.gpus

    test_model(setting_dict, args.checkpoint)

