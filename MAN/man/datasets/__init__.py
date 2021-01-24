from man.datasets.mini_imagenet import MiniImageNet
from man.datasets.mini_imagenet import MiniImageNet3x3Patches
from man.datasets.mini_imagenet import MiniImageNet80x80
from man.datasets.mini_imagenet import MiniImageNetImagesAnd3x3Patches


all_datasets = dict(
    MiniImageNet=MiniImageNet,
    MiniImageNet80x80=MiniImageNet80x80,
    MiniImageNet3x3Patches=MiniImageNet3x3Patches,
    MiniImageNetImagesAnd3x3Patches=MiniImageNetImagesAnd3x3Patches,
)


def dataset_factory(dataset_name, *args, **kwargs):
    return all_datasets[dataset_name](*args, **kwargs)
