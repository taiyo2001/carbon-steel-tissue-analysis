import os

from skimage import io, transform  # scikit-image
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as alb
from albumentations.pytorch import ToTensorV2


# 画像データ拡張の関数
def get_train_transform(
    image_height, image_width, horizontal_flip=0.25, vertical_flip=0.25
):
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
            ToTensorV2(),
        ]
    )


# Datasetクラスの定義
class LoadDataSet(Dataset):
    def __init__(self, path, image_height, image_width, transform=None):
        self.path = path
        folders = os.listdir(path)
        self.folders = sorted(folders, key=lambda x: tuple(map(int, x.split("-"))))
        self.image_height = image_height
        self.image_width = image_width
        self.transforms = transform  # get_train_transform(256, 256)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        image_folder = os.path.join(self.path, self.folders[idx], "images/")
        mask_folder = os.path.join(self.path, self.folders[idx], "masks/")
        image_path = os.path.join(image_folder, os.listdir(image_folder)[0])

        # debug
        # print(f'image_folder: {image_folder}')
        # print(f'mask_folder: {mask_folder}')
        # print(f'image_path: {image_path}')
        # print(f'mask_path: {mask_path}')

        # 画像データの取得
        # img = io.imread(image_path)[:,:,:3].astype('float32')
        img = io.imread(image_path)
        # TODO: 正答率がでたら比較としてfloat32でも試してみる
        img = self.conv_2D_to_3Darray(img)

        # maskの量が2つ以上になったらここの関数を実行してすべてを得る
        # mask = self.get_mask(mask_folder, 256, 256).astype('float32')
        mask = self.get_mask(
            mask_folder, self.image_height, self.image_width
        )  # nint8でOK?

        # 前処理をするためにひとつにまとめる
        augmented = self.transforms(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]
        return (img, mask)

    def conv_2D_to_3Darray(self, arr):
        image = Image.fromarray(arr)
        image = image.convert("RGB")
        # 変換するならここでfloat32にする
        arr = np.asarray(image, np.uint8)
        return arr

    def conv_3D_to_2Darray(self, arr):
        image = Image.fromarray(arr)
        image = image.convert("L")
        arr = np.asarray(image, np.uint8)  # 変換するならここでfloat32にする
        return arr

    # マスクデータの取得
    def get_mask(self, mask_folder, IMG_HEIGHT, IMG_WIDTH):
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
        # print(f'len(os.listdir(mask_folder)): {len(os.listdir(mask_folder))}')
        for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(os.path.join(mask_folder, mask_))
            # RGBからLへ
            mask_ = self.conv_3D_to_2Darray(mask_)
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_ = np.expand_dims(mask_, axis=-1)
            # mask = np.maximum(mask, mask_)
            mask = mask_

        return mask
