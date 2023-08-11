from module import const
import os

# 定数の設定
const.APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

const.TRAIN_PATH = const.APP_PATH + '/data/img'
const.CHECKPOINT_PATH = const.APP_PATH + '/data/model'
