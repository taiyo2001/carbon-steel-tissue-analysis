# carbon-steel-tissue-analysis
深層学習によるフェライト・パーライト組織のセグメンテーションを用いた組織解析

Analysis of ferrite-pearlite microstructure using deep learning segmentation

## フォルダー構造
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" integrity="sha384-9JzIyL4wS5E9T6XvfD0ySc1JC63B4r1mOtL4fFv3Eg==" crossorigin="anonymous">
<pre>

.
├── 🗁 app
│   ├── 🗁 services
│   │   ├── ![SVG Icon](/readme_svg/python.svg) create_mask_image.ipynb
│   │   ├── ![SVG Icon](/readme_svg/python.svg) image_analysis.ipynb
│   │   ├── ![SVG Icon](/readme_svg/python.svg) image_analysis_confirm.ipynb
│   │   └── ![SVG Icon](/readme_svg/python.svg) rough_and_fine_image_analysis.ipynb
│   ├── ![SVG Icon](/readme_svg/python.svg) unet_colab.ipynb
│   └── ![SVG Icon](/readme_svg/python.svg) unet_command.ipynb
├── 🗁 bin
│   ├── ![SVG Icon](/readme_svg/terminal_shell.svg) docker_setup.sh
│   ├── ![SVG Icon](/readme_svg/terminal_shell.svg) exec_ipynb.sh
│   ├── ![SVG Icon](/readme_svg/terminal_shell.svg) formatter.sh
│   └── ![SVG Icon](/readme_svg/terminal_shell.svg) setup.sh
├── 🗁 config
│   ├── ![SVG Icon](/readme_svg/python.svg) __init__.py
│   └── ![SVG Icon](/readme_svg/python.svg) setting.py
├── 🗁 data
│   └── 🗁 model
│       ├── 🗁 SegNet
│       └── 🗁 UNet
├── 🗁 module
│   ├── ![SVG Icon](/readme_svg/python.svg) __init__.py
│   ├── ![SVG Icon](/readme_svg/python.svg) const.py
│   ├── ![SVG Icon](/readme_svg/python.svg) image_loader.py
│   └── ![SVG Icon](/readme_svg/python.svg) tissue_analysis.py
├── 🗁 readme_svg
│   ├── docker.svg
│   ├── md_file.svg
│   ├── python.svg
│   ├── terminal_shell.svg
│   └── text.svg
├── 🗁 result
│   ├── 🗁 services
│   │   └── ![SVG Icon](/readme_svg/python.svg) rough_and_fine_image_analysis_epoch_9px.nbconvert.ipynb
│   └── ![SVG Icon](/readme_svg/python.svg) unet_command_result.nbconvert.ipynb
├── ![SVG Icon](/readme_svg/docker.svg) Dockerfile
├── ![SVG Icon](/readme_svg/md_file.svg) README.md
├── ![SVG Icon](/readme_svg/text.svg) requirements.sample.txt
├── ![SVG Icon](/readme_svg/text.svg) requirements.txt
└── ![SVG Icon](/readme_svg/python.svg) setup.py
</pre>
13 directories, 28 files

## 環境構築
機械学習ライブラリにPyTorchを使用しているためGPUが必須.

### データ準備

`/data`直下に以下のような構造でimgディレクトリを用意する.
<pre>
├── 🗁 data
│   ├── 🗁 img(`img`には`config/setting.py`内の`const.TRAIN_DIR`の値を入れる)
│   │   ├── 🗁 images ─...
│   │   └── 🗁 masks  ─...
│   └─── 🗁 model
│       ├── 🗁 SegNet
│       └── 🗁 UNet
</pre>

### Docker
Build & Running Docker Container
```
sh bin/docker_setup.sh
```
コンテナ内に入ったらaliasを登録するために毎回以下を実行.
```
source bin/setup.sh
```

## Jupyter Notebookの実行方法
`/app`内のJupyter Notebookを実行するときは、APP_PATHで以下を叩くと実行結果ファイル`/result/${file_name}_epoch_${epochs}.nbconvert.ipynb`が出力される.
```
ipynb ${file_name} ${epochs}
```
※第2引数は省略可

ex. 300epochs分を回す`/app/unet_command.ipynb`を実行時は以下を実行.
```
ipynb unet_command 300
```
出力ファイル：`/result/unet_command_epoch_300.nbconvert.ipynb`

## フォーマットの実行方法
Check Linter
```
black ./ --check
```

Exec Linter
```
black ./
```
