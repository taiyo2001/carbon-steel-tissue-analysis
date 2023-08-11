# carbon-steel-tissue-analysis
フェライト・パーライト組織を有する炭素鋼の深層学習によるセグメンテーション

## 研究目的
深層学習によるセグメンテーションによってフェライト・パーライト組織を有する炭素鋼の組織識別から組織情報を抽出する.

## フォルダー構造
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" integrity="sha384-9JzIyL4wS5E9T6XvfD0ySc1JC63B4r1mOtL4fFv3Eg==" crossorigin="anonymous">
<pre>

.
├── 🗁 app
│   ├── ![SVG Icon](python.svg) unet_colab.ipynb
│   └── ![SVG Icon](python.svg) unet_command.ipynb
├── 🗁 result
│   └── unet_command_result.nbconvert.ipynb
├── 🗁 bin
│   └── ![SVG Icon](python.svg) setup.py
├── 🗁 config
│   ├── ![SVG Icon](python.svg) \_\_init__.py
│   └── ![SVG Icon](python.svg) setting.py
├── 🗁 data
│   └── 🗁 model
│       ├── 🗁 SegNet
│       └── 🗁 UNet
├── 🗁 module
│   ├── ![SVG Icon](python.svg) const.py
│   ├── ![SVG Icon](python.svg) image_loader.py
│   └── ![SVG Icon](python.svg) \_\_init__.py
├── ![SVG Icon](docker.svg) Dockerfile
├──![SVG Icon](md_file.svg)README.md
├── ![SVG Icon](text.svg) requirements.txt
├── ![SVG Icon](terminal_shell.svg) exec_ipynb.sh
└── ![SVG Icon](terminal_shell.svg) setup.sh
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
