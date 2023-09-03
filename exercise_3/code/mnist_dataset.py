import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from code.boiler_plate.dataset import Dataset


class MNISTDataset(Dataset):
    def __init__(
        self,
        root,
        download=False,
        load_to_memory=False,
        download_urls=[
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        ],
        dowload_prefix="train",
        shuffle=True,
    ):
        self.root_path = root
        self.dowload_prefix = dowload_prefix
        self.load_to_memory = load_to_memory

        self.length = None

        if download:
            for url in download_urls:
                super().__init__(root, download, url)
                self._convert_data(dowload_prefix)

        if self.load_to_memory:
            self.images, self.labels = self._load_data(dowload_prefix)

        self.indices = list(range(self.number_of_images()))
        if shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        if self.load_to_memory:
            return self.images[self.indices[index]], self.labels[self.indices[index]]
        else:
            img = Image.open(
                os.path.join(
                    self.root_path,
                    f"{self.dowload_prefix}_images/{self.indices[index]}.png",
                )
            )
            with open(
                os.path.join(
                    self.root_path,
                    f"{self.dowload_prefix}_labels/{self.indices[index]}.txt",
                ),
                "r",
            ) as f:
                label = int(f.read())
            return np.array(img), label

    def __len__(self):
        return self.number_of_images()

    def number_of_images(self):
        if self.load_to_memory:
            return len(self.images)
        else:
            if self.length == None:
                self.length = len(
                    os.listdir(
                        os.path.join(self.root_path, f"{self.dowload_prefix}_images")
                    )
                )
            return self.length

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self):
            raise StopIteration
        img, label = self.__getitem__(self.current_index)
        self.current_index += 1
        return img, label

    def show_image(self, index):
        image, label = self.__getitem__(index)
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label}")
        plt.show()

    def _load_data(self, prefix):
        images_folder = os.path.join(self.root_path, f"{prefix}_images")
        labels_folder = os.path.join(self.root_path, f"{prefix}_labels")

        images = []
        labels = []
        for i in range(len(os.listdir(images_folder))):
            img = Image.open(os.path.join(images_folder, f"{i}.png"))
            images.append(np.array(img))
            with open(os.path.join(labels_folder, f"{i}.txt"), "r") as f:
                labels.append(int(f.read()))

        return np.array(images), np.array(labels)

    # NOT INTERESTING
    def _convert_data(self, prefix):
        # Convert the MNIST dataset to PNG images and text labels
        images_folder = os.path.join(self.root_path, f"{prefix}_images")
        labels_folder = os.path.join(self.root_path, f"{prefix}_labels")
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)

        with open(
            os.path.join(self.root_path, f"{prefix}-images-idx3-ubyte"), "rb"
        ) as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        with open(
            os.path.join(self.root_path, f"{prefix}-labels-idx1-ubyte"), "rb"
        ) as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        for i, (img, label) in enumerate(zip(images, labels)):
            img = Image.fromarray(img, mode="L")
            img.save(os.path.join(images_folder, f"{i}.png"))
            with open(os.path.join(labels_folder, f"{i}.txt"), "w") as f:
                f.write(str(label))
