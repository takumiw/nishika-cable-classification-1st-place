# Nishika Cable Connector Classification Challenge
> [Nishika ケーブルコネクタの種類判別コンペティション](https://www.nishika.com/competitions/19/summary)の1位解法と学習コードです。  
> 解法は[こちら](https://github.com/takumiw/nishika-cable-classification-1st-place/blob/main/colution.md)と、[こちら](https://www.nishika.com/competitions/19/topics)のディスカッションに同じものを記載しています。

# Directory Layout
```
.
├── configs
│    ├── exp
│    │    ├── exp00.yaml
│    │    ├── ...
│    │    └── exp99.yaml
│    ├── config.yaml
│    └── settings.py
├── img
├── input 
│    ├── preprocessed
│    │    ├── test_1280
│    │    ├── test_all_clean_cropped
│    │    ├── test_all_clean_cropped_pad1
│    │    ├── train_all_clean_1280
│    │    ├── train_all_clean_cropped
│    │    └── train_all_clean_cropped_pad1
│    ├── train
│    │    ├── 000c15d0.jpg
│    │    ├── ...
│    │    └── ffe8cdb5.jpg
│    ├── additional
│    │    ├── 00037f39.jpg
│    │    ├── ...
│    │    └── ffd560d1.jpg
│    ├── test
│    │    ├── 0055471d.jpg
│    │    ├── ...
│    │    └── ffeace36.jpg
│    ├── yolo
│    ├── train_all
│    ├── additional_data.csv
│    ├── sample_submission.csv
│    ├── train_all_clean.csv
│    ├── train_all_clean_Lightning_T_all.csv
│    └── train.csv
├── logs 
├── outputs
├── src
│    ├── data_vis
│    ├── models
│    ├── postprocess
│    ├── preprocess
│    ├── ...
│    └── utils.py
├── tmp
├── .env.sample
├── README.md
├── requirements.txt
├── requirements_cpu.txt
├── run_00_preparation_00.sh
├── ...
└── run_12_stage3_avg_exp36.sh
```

# Requirements
下記の環境で動作確認済みです。
- Python 3.8.7
- GPU: NVIDIA V100
- cuda 11.1.1
- cudnn 8.2.1
- gcc 9.3.0

利用したパッケージは、`requirements.txt`に記載してあります

# How to Run
## データのダウンロード
1. https://www.nishika.com/competitions/19/data からコンペのデータをダウンロードし、上記のようにinputディレクトリに展開してください。

2. 必要なパッケージをインストールしてください。

```
$ python -m venv venv
$ source venv/bin/activate
(venv) $ pip install --upgrade pip

# GPUがある場合
(venv) $ pip install -r requirements.txt

# CPUだけの場合
(venv) $ pip install -r requirements_cpu.txt
```

## 各種実行
`run_00_preparation_00.sh` から順に `run_12_stage3_avg_exp36.sh` まで実行してください。  
実行結果は `logs` ディレクトリに自動で保存されます。`run_11_stage3_avg_exp30.sh`、`run_12_stage3_avg_exp36.sh`の実行後 `logs`ディレクトリに出力される `submission.csv` ファイルが、最終提出に利用した提出ファイルとなります。  
(注. 上記を全て直列に実行した場合、合計100時間程度かかります。)