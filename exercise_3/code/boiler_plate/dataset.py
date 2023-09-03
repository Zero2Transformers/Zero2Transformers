from abc import ABC, abstractmethod
import os
import urllib.request
import tarfile
import shutil
import gzip


class Dataset(ABC):
    def __init__(self, root, download=False, download_url=None):
        self.root_path = root

        if download:
            if download_url is None:
                raise ValueError(
                    "download_url must be provided when download is set to True"
                )
            self._download_dataset(download_url)
        else:
            if not self._check_root_directory():
                raise FileNotFoundError(
                    f"Root directory '{self.root_path}' does not exist."
                )

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    # NOT INTERESTING
    def _download_dataset(self, url):
        print("Downloading dataset...")
        os.makedirs(self.root_path, exist_ok=True)
        filename = url.split("/")[-1]
        download_path = os.path.join(self.root_path, filename)

        try:
            urllib.request.urlretrieve(url, download_path, self._progress_callback)
            print("\nDownload completed.")
            if filename.endswith(".gz"):
                with gzip.open(download_path, "rb") as f_in:
                    with open(download_path[:-3], "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(download_path)
                print("Dataset extracted.")
            elif filename.endswith(".tar.gz"):
                with tarfile.open(download_path, "r:gz") as tar:
                    tar.extractall(self.root_path)
                os.remove(download_path)
                print("Dataset extracted.")
            else:
                print("Unknown file type.")
        except Exception as e:
            print(f"Download failed: {e}")

    # NOT INTERESTING
    def _progress_callback(self, block_num, block_size, total_size):
        downloaded_bytes = block_num * block_size
        total_mb = total_size / (1024 * 1024)  # Convert total_size to MB
        downloaded_mb = downloaded_bytes / (
            1024 * 1024
        )  # Convert downloaded_bytes to MB

        percentage = min(downloaded_bytes / total_size, 1.0)
        progress = int(percentage * 100)
        print(
            f"Download progress: {progress}% ({downloaded_mb:.2f}/{total_mb:.2f} MB)",
            end="\r",
        )

    def _check_root_directory(self):
        return os.path.exists(self.root_path)
