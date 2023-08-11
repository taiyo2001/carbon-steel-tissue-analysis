# carbon-steel-tissue-analysis
フェライト・パーライト組織を有する炭素鋼の深層学習によるセグメンテーション

## 研究目的
深層学習によるセグメンテーションによってフェライト・パーライト組織を有する炭素鋼の組織識別から組織情報を抽出する.

## フォルダー構造
<pre>
.
├── app
│   ├── unet_colab.ipynb
│   └── unet_command.ipynb
├── result
│   └── unet_command_result.nbconvert.ipynb
├── bin
│   └── setup.py
├── config
│   ├── __init__.py
│   └── setting.py
├── data
│   └── model
│       ├── SegNet
│       └── UNet
├── module
│   ├── const.py
│   ├── image_loader.py
│   └── __init__.py
├── Dockerfile
├── README.md
├── requirements.txt
├── exec_ipynb.sh
└── setup.sh

</pre>
9 directories, 14 files

## 環境構築
### Docker

データの準備

`/data`直下に以下のような構造でimgディレクトリを用意する
<pre>
├── data
│   ├── img
│   │   ├── images ─...
│   │   └── masks  ─...
│   └─── model
│       ├── SegNet
│       └── UNet
</pre>

Build
```
docker build -t [name] .
```

立ち上げ
```
docker start [name]
docker container exec -it [name] bash
cd /root
source setup.sh
```

終了
```
exit
docker stop [name]
```

### Local
`.tool_versions`でpythonのバージョンを確認し、各自で構築.

## shによるJupyter Notebookの実行方法
`/app`内のJupyter Notebookを実行するときは、APP_PATHで以下を叩くと実行結果ファイル`/result/${file name}_epoch_${epochs}.nbconvert.ipynb`が出力される.
```
sh exec_ipynb.sh ${file name} ${epochs}
```

ex. 300epochs分を回す`/app/unet_command.ipynb`を実行するとき

実行コマンド
```
sh exec_ipynb.sh unet_command 300
```
出力ファイル：`/result/unet_command_epoch_300.nbconvert.ipynb`
