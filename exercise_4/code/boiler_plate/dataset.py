from abc import ABC, abstractmethod
import os
import urllib.request
import tarfile
import shutil
import gzip


class Dataset(ABC):
    """
    Abstract Base Class for Dataset handling.

    This class serves as a template for building datasets.
    It includes functionalities for downloading data and
    checking the presence of root directories.

    Attributes:
        root_path (str): Path to the root directory where the data is or will be stored.
        data: The actual dataset (could be of any type).

    Methods:
        _download_dataset: Downloads the dataset.
        _check_root_directory: Checks if the root directory exists.
    """

    def __init__(self, root=None, download=False, download_url=None, data=None):
        """
        Initializes the Dataset object.

        Args:
            root (str, optional): Path to the root directory.
            download (bool, optional): Whether to download the dataset. Default is False.
            download_url (str, optional): The URL from which the dataset will be downloaded.
            data: The actual dataset if provided.

        Raises:
            ValueError: If there are contradictions in the provided arguments.
            FileNotFoundError: If the specified root directory doesn't exist.
        """

        self.root_path = root

        # Ensuring proper arguments
        if root is None and data is None:
            raise ValueError(
                "Provide either the data or the filepath to the data folder."
            )

        if data is not None and download:
            raise ValueError(
                "Can't download if data is directly passed as a parameter."
            )

        # Handling data attribute
        if data is not None:
            self.data = data
            return

        # Handling download
        if download:
            if not download_url:
                raise ValueError(
                    "A download URL must be provided when download is set to True."
                )
            self._download_dataset(download_url)

        # Checking the root directory
        elif not self._check_root_directory():
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
