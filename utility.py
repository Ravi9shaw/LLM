import numpy as np
from pathlib import Path
import pandas as pd
import tarfile
import zipfile
import urllib.request as req
from zlib import crc32


def download_and_extract(url, root_dir="datasets"):
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    file_name = url.split("/")[-1]
    file_path = root / file_name  

    if not file_path.exists():
        req.urlretrieve(url, file_path)

    if file_name.endswith((".tgz", ".tar.gz")):
        with tarfile.open(file_path) as tar:
            tar.extractall(path=root)

    elif file_name.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(root)

    csv_files = list(root.rglob("*.csv"))
    if csv_files:
        return pd.read_csv(csv_files[0])

    return None
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32
def split_train_test_by_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
