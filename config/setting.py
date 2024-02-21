from module import const
import os

# 定数の設定
const.APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# const.TRAIN_DIR = '/img'       # S45Cのみの使用
# const.TRAIN_DIR = '/img_rough'  # S10C, S15C. S45Cのあるrough maskを使用
const.TRAIN_DIR = "/img_fine"  # S10C, S15C. S45Cのあるfine maskを使用

const.TRAIN_PATH = const.APP_PATH + "/data" + const.TRAIN_DIR
const.CHECKPOINT_PATH = const.APP_PATH + "/data/model"
