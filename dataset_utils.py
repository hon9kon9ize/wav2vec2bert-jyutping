from pathlib import Path


DEFAULT_DATASET = "indiejoseph/tts20250516"


def load_any_dataset(dataset: str = DEFAULT_DATASET):
    from datasets import load_dataset, load_from_disk

    if Path(dataset).exists():
        return load_from_disk(dataset)

    return load_dataset(dataset)


def flatten_dataset(dataset):
    from datasets import DatasetDict, concatenate_datasets

    if isinstance(dataset, DatasetDict):
        return concatenate_datasets([dataset[split] for split in dataset])

    return dataset


def make_train_validation_test_split(
    dataset,
    test_size: int = 500,
    validation_size: int = 500,
    seed: int = 42,
):
    from datasets import DatasetDict

    dataset = flatten_dataset(dataset).shuffle(seed=seed)
    total_size = len(dataset)
    test_end = min(test_size, total_size)
    validation_end = min(test_end + validation_size, total_size)

    return DatasetDict(
        {
            "test": dataset.select(range(0, test_end)),
            "validation": dataset.select(range(test_end, validation_end)),
            "train": dataset.select(range(validation_end, total_size)),
        }
    )
