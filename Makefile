# インストールされていなかったら指定のバージョンをインストールする
# ビルド系はbin/setupなどに分けたほうが管理しやすい？？
# (Makefileはコンパイル作業)
package:
	python -m pip install --upgrade pip
