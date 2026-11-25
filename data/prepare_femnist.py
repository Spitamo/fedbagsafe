import gc
import tarfile
from pathlib import Path
from urllib.request import urlopen

import h5py
import numpy as np
import torch
from tqdm import tqdm

FEMNIST_URL = "https://fedml.s3-us-west-1.amazonaws.com/fed_emnist.tar.bz2"


def _load_h5_dataset(file_path: Path):
    """load images and labels from hdf5 file"""
    with h5py.File(file_path, "r") as f:
        client_ids = list(f["examples"].keys())
        all_data, all_labels = [], []

        for cid in tqdm(client_ids, desc=f"Loading {file_path.name}"):
            pixels = f["examples"][cid]["pixels"][()].astype(np.float32)
            labels = f["examples"][cid]["label"][()].astype(np.int64)
            all_data.append(pixels)
            all_labels.append(labels)

    data_np = np.vstack(all_data)[:, np.newaxis, :, :]
    labels_np = np.hstack(all_labels)
    return torch.from_numpy(data_np), torch.from_numpy(labels_np)


def _download_file(url: str, destination: Path):
    """download archive file"""
    with urlopen(url) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        with open(destination, "wb") as f, tqdm(
            total=total_size or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading FEMNIST",
        ) as progress:
            while chunk := response.read(1024 * 1024):
                f.write(chunk)
                progress.update(len(chunk))


def _extract_archive(archive: Path, output_dir: Path):
    """extract tar.bz2 archive"""
    with tarfile.open(archive, "r:bz2") as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc="Extracting FEMNIST", unit="file") as progress:
            for member in members:
                tar.extract(member, path=output_dir, filter="data")
                progress.update(1)


def prepare_flattened_emnist(data_dir: str | Path) -> Path:
    """download and cache flattened femnist dataset"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    archive = data_dir / "fed_emnist.tar.bz2"
    train_file = data_dir / "fed_emnist_train.h5"
    test_file = data_dir / "fed_emnist_test.h5"
    cache_file = data_dir / "flattened_emnist.pt"

    if cache_file.exists():
        return cache_file

    if not (train_file.exists() and test_file.exists()):
        try:
            _download_file(FEMNIST_URL, archive)
            _extract_archive(archive, data_dir)
        finally:
            archive.unlink(missing_ok=True)

    for file in (train_file, test_file):
        if not file.exists():
            raise FileNotFoundError(f"Missing dataset file: {file}")

    train_data, train_labels = _load_h5_dataset(train_file)
    gc.collect()
    test_data, test_labels = _load_h5_dataset(test_file)
    gc.collect()

    torch.save(
        {
            "train_data": train_data,
            "train_labels": train_labels,
            "test_data": test_data,
            "test_labels": test_labels,
        },
        cache_file,
    )
    return cache_file