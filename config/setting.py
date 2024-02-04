from module import const
import os

# 定数の設定
const.APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# const.TRAIN_DIR = '/img'       # 45のみの使用
# const.TRAIN_DIR = '/img_full'  # 10, 15. 45のすべてを使用
const.TRAIN_DIR = "/img_new"  # 新たに作成したものを使用

const.TRAIN_PATH = const.APP_PATH + "/data" + const.TRAIN_DIR
const.CHECKPOINT_PATH = const.APP_PATH + "/data/model"
