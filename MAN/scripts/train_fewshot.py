"""Tests a few-shot recognition models.

Example of usage:
# Train the CC+Rot model on MiniImageNet, run
$ python scripts/train_fewshot.py --config=miniImageNet/WRNd28w10CosineClassifierRotAugRotSelfsupervision

The config argument specifies the model that will be trained.
"""


import argparse
import os

from man import project_root
from man.algorithms.fewshot.fewshot import (
    FewShot,
)
from man.algorithms.mix_selfsupervision.fewshot_selfsupervision_rot_loc_jig_clu_att_fc import (
    FewShotRotLocJigCluSelfSupervisionAttFC,
)
from man.dataloaders.basic_dataloaders import OnlyImageDataloader
from man.dataloaders.dataloader_fewshot import FewShotDataloader
from man.dataloaders.dataloader_fewshot import FewShotDataloaderWithPatches
from man.dataloaders.multiple_dataloaders import TwoParallelDataloaders
from man.datasets import all_datasets
import pdb
print(project_root)
def get_arguments():
    """ Parse input arguments. """
    parser = argparse.ArgumentParser(
        description="Code for training few-shot image recogntion models."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="",
        help="config file with parameters of the experiment.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=0,
        help="checkpoint (epoch id) that will be loaded. If a negative value "
        "is given then the latest existing checkpoint is loaded.",
    )
    parser.add_argument("--cuda", type=bool, default=True, help="enables cuda.")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of workers for the data loader(s).",
    )
    parser.add_argument(
        "--disp_step",
        type=int,
        default=200,
        help="how frequently the progress of the training will be printed.",
    )
    parser.add_argument(
        "--k_number",
        type=int,
        default=10,
        help="the number of center when using cluster",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=640,
        help="the number of center when using cluster",
    )
    args_opt = parser.parse_args()
    exp_base_directory = os.path.join(project_root, "experiments")
    exp_directory = os.path.join(exp_base_directory, args_opt.config)

    # Load the configuration params of the experiment
    exp_config_file = "config." + args_opt.config.replace("/", ".")
    print(f"Loading experiment {args_opt.config}")
    config = __import__(exp_config_file, fromlist=[""]).config
    config["exp_dir"] = exp_directory  # where logs, models, etc will be stored.
    print(f"Generated logs and/or snapshots will be stored on {exp_directory}")

    return args_opt, exp_directory, config


def main():
    args_opt, exp_directory, config = get_arguments()

    # Set train and test datasets and the corresponding data loaders
    data_train_opt = config["data_train_opt"]
    data_test_opt = config["data_test_opt"]

    if "dataset_args" in data_train_opt:
        dataset_train_args = data_train_opt["dataset_args"]
    else:
        dataset_train_args = {"phase": data_train_opt["phase"]}

    if "dataset_args" in data_test_opt:
        dataset_val_args = data_test_opt["dataset_args"]
    else:
        dataset_val_args = {"phase": "val"}

    train_dataset_class = all_datasets[data_train_opt["dataset_name"]]
    dataset_train = train_dataset_class(**dataset_train_args)
    val_dataset_class = all_datasets[data_test_opt["dataset_name"]]
    dataset_val = val_dataset_class(**dataset_val_args)

    algorithm_type = config.get("algorithm_type")
    if algorithm_type == "mix_selfsupervision.fewshot_selfsupervision_rot_loc_jig_clu_att_fc" :
        data_train_patches_opt = config["data_train_patches_opt"]
        dataset_train_patches_args = data_train_patches_opt["dataset_args"]
        class_dataset = all_datasets[data_train_patches_opt["dataset_name"]]
        dataset_train_patches = class_dataset(**dataset_train_patches_args)
        dloader_train = FewShotDataloaderWithPatches(
            dataset=dataset_train,
            dataset_patches=dataset_train_patches,
            nKnovel=data_train_opt["nKnovel"],
            nKbase=data_train_opt["nKbase"],
            n_exemplars=data_train_opt["n_exemplars"],
            n_test_novel=data_train_opt["n_test_novel"],
            n_test_base=data_train_opt["n_test_base"],
            batch_size=data_train_opt["batch_size"],
            num_workers=args_opt.num_workers,
            epoch_size=data_train_opt["epoch_size"],
        )
    else:
        dloader_train = FewShotDataloader(
            dataset=dataset_train,
            nKnovel=data_train_opt["nKnovel"],
            nKbase=data_train_opt["nKbase"],
            n_exemplars=data_train_opt["n_exemplars"],
            n_test_novel=data_train_opt["n_test_novel"],
            n_test_base=data_train_opt["n_test_base"],
            batch_size=data_train_opt["batch_size"],
            num_workers=args_opt.num_workers,
            epoch_size=data_train_opt["epoch_size"],
        )
    #pdb.set_trace()
    if config.get("semi_supervised") is True:
        data_unlabeled_opt = config["data_unlabeled_opt"]
        class_dataset_unlabeled = all_datasets[data_unlabeled_opt["dataset_name"]]
        dataset_unlabeled = class_dataset_unlabeled(phase=data_unlabeled_opt["phase"])

        dloader_unlabeled = OnlyImageDataloader(
            dataset=dataset_unlabeled,
            batch_size=data_unlabeled_opt["batch_size"],
            num_workers=args_opt.num_workers,
            epoch_size=data_unlabeled_opt["epoch_size"],
            train=True,
        )

        dloader_train = TwoParallelDataloaders(dloader_train, dloader_unlabeled)

    dloader_val = FewShotDataloader(
        dataset=dataset_val,
        nKnovel=data_test_opt["nKnovel"],
        nKbase=data_test_opt["nKbase"],
        n_exemplars=data_test_opt["n_exemplars"],
        n_test_novel=data_test_opt["n_test_novel"],
        n_test_base=data_test_opt["n_test_base"],
        batch_size=data_test_opt["batch_size"],
        num_workers=args_opt.num_workers,
        epoch_size=data_test_opt["epoch_size"],
    )

    config["disp_step"] = args_opt.disp_step
    if algorithm_type == "fewshot.fewshot":
        algorithm = FewShot(config)
    elif algorithm_type == "mix_selfsupervision.fewshot_selfsupervision_rot_loc_jig_clu_att_fc":
        print(args_opt.k_number)
        algorithm = FewShotRotLocJigCluSelfSupervisionAttFC(config, k=args_opt.k_number, w = args_opt.w)

    if args_opt.cuda:  # enable cuda
        algorithm.load_to_gpu()

    if args_opt.checkpoint != 0:  # load checkpoint
        algorithm.load_checkpoint(
            epoch=args_opt.checkpoint if (args_opt.checkpoint > 0) else "*", train=True
        )

    # train the algorithm
    algorithm.solve(dloader_train, dloader_val)
        


if __name__ == "__main__":
    main()
