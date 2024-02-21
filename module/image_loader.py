import os

from skimage import io, transform
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import albumentations as alb
from typing import Optional, Tuple

def get_train_transform(
    image_height: int, image_width: int, horizontal_flip: float = 0.25, vertical_flip: float = 0.25
) -> alb.Compose:
    """
    Returns the image data augmentation transformation for training.

    Parameters:
    - image_height (int): Height of the resized image.
    - image_width (int): Width of the resized image.
    - horizontal_flip (float): Probability of horizontal flipping (default is 0.25).
    - vertical_flip (float): Probability of vertical flipping (default is 0.25).

    Returns:
    alb.Compose: Augmentation pipeline for training data.

    This function returns an augmentation pipeline using Albumentations library.
    The pipeline consists of the following transformations:
    - Resize the image to the specified height and width.
    - Normalize the image with mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).
    - Apply horizontal flipping with the given probability.
    - Apply vertical flipping with the given probability.
    - Convert the image to PyTorch tensor format.
    """

    return alb.Compose(
        [
            # リサイズ(元画像ですでにしているが)
            alb.Resize(image_height, image_width),
            # 正規化(こちらの細かい値はalbumentations.augmentations.transforms.Normalizeのデフォルトの値を適用)
            alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # 水平フリップ（pはフリップする確率）
            alb.HorizontalFlip(p=horizontal_flip),
            # 垂直フリップ
            alb.VerticalFlip(p=vertical_flip),
            alb.pytorch.ToTensorV2(),
        ]
    )

class LoadDataSet(Dataset):
    """
    Dataset class for loading images and masks.

    Parameters:
    - path (str): Path to the dataset folder.
    - image_height (int): Height of the resized image.
    - image_width (int): Width of the resized image.
    - transform (optional): Image transformation pipeline (default is None).

    Returns:
    Tuple[np.ndarray, np.ndarray]: Tuple containing the image and its corresponding mask.

    This class loads images and masks from the specified dataset folder.
    """

    def __init__(
        self,
        path: str,
        image_height: int,
        image_width: int,
        transform: Optional[callable] = None
    ) -> None:
        self.path = path
        folders = os.listdir(path)
        self.folders = sorted(folders, key=lambda x: tuple(map(int, x.split("-"))))
        self.image_height = image_height
        self.image_width = image_width
        self.transforms = transform  # get_train_transform(256, 256)

    def __len__(self) -> int:
        return len(self.folders)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the image and mask at the specified index.

        Parameters:
        - idx (int): Index of the data item.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the image and its corresponding mask.
        """
        image_folder = os.path.join(self.path, self.folders[idx], "images/")
        mask_folder = os.path.join(self.path, self.folders[idx], "masks/")
        image_path = os.path.join(image_folder, os.listdir(image_folder)[0])

        img = io.imread(image_path)
        img = self.conv_2D_to_3Darray(img)

        mask = self.get_mask(
            mask_folder, self.image_height, self.image_width
        )

        # 前処理をするためにひとつにまとめる
        augmented = self.transforms(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]
        return (img, mask)

    def conv_2D_to_3Darray(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert 2D array to 3D array.

        Parameters:
        - arr (np.ndarray): Input 2D array.

        Returns:
        np.ndarray: Converted 3D array.
        """
        image = Image.fromarray(arr)
        image = image.convert("RGB")
        arr = np.asarray(image, np.uint8)
        return arr

    def conv_3D_to_2Darray(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert 3D array to 2D array.

        Parameters:
        - arr (np.ndarray): Input 3D array.

        Returns:
        np.ndarray: Converted 2D array.
        """
        image = Image.fromarray(arr)
        image = image.convert("L")
        arr = np.asarray(image, np.uint8)
        return arr

    def get_mask(self, mask_folder: str, IMG_HEIGHT: int, IMG_WIDTH: int) -> np.ndarray:
        """
        Get the mask data.

        Parameters:
        - mask_folder (str): Path to the mask folder.
        - IMG_HEIGHT (int): Height of the image.
        - IMG_WIDTH (int): Width of the image.

        Returns:
        np.ndarray: Mask data.
        """
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
        # print(f'len(os.listdir(mask_folder)): {len(os.listdir(mask_folder))}')
        for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(os.path.join(mask_folder, mask_))
            mask_ = self.conv_3D_to_2Darray(mask_)
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = mask_

        return mask
