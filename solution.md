みなさまお疲れ様でした！！！    
入賞候補精査中ではありますが、1位に入りましたので取り急ぎ解法や取り組みを投稿します。ご質問やご指摘などございましたが、コメントいただければと思います。  
最終submissionのコードはこちらで公開予定です。  
https://github.com/takumiw/nishika-cable-classification-1st-place

# 3行まとめ
- YOLOv5でケーブルコネクタのObject Detectionを行い、検出範囲で画像をクロッピング
- クロッピングした画像を使い、EfficientNet、ResNetなどで分類
- AveragingやStackingなどで複数モデルをアンサンブル

# 解法
## 前処理
### 物体検出用アノテーションデータ
今回提供された訓練用データセットは、`train`(2371枚)と`additional`(1648枚)の2種類ありました。  
コンペ中、[以下のアナウンス](https://www.nishika.com/competitions/19/topics/143)が行われたため、additionalデータセットの画像全てに物体検出用のアノテーションデータを手作業で追加しました。また、`train`データセットのアノテーションデータもノイズが大きかったため、手作業で修正しました。アノテーションツールには[labelImg](https://github.com/tzutalin/labelImg)を使っています。

<img src="https://raw.githubusercontent.com/takumiw/nishika-cable-classification-1st-place/main/img/img0_1.png" width=90%>

### 画像分類用アノテーションデータ
今回提供された訓練用データセットには、画像に実際に写っているケーブルコネクタとは異なるラベルが付与されていることが多々あったため、手作業で修正しました (私が確認できたもので、29枚ありました)。また、`bf31e612.jpg`, `65bba5fb.jpg`, `94cfdb2c.jpg`, `66945e57.jpg`の合計4枚の画像は学習に悪影響を及ぼす可能性を考え、除外しました (目見でもクラス判別ができなかったり、ケーブルコネクタ全体が写っていないなど)。  
なお、以上の方法で前処理を施したアノテーションデータは、[こちら](https://github.com/takumiw/nishika-cable-classification-1st-place/blob/main/input/train_all_clean.csv)で公開しています。(アノテーションデータはYOLOv5のフォーマットです。)

## Stage-1: YOLOv5でケーブルコネクタのObject Detection
### YOLOv5の学習
前処理を施したデータセットを使い、[YOLOv5](https://github.com/ultralytics/yolov5)で学習を行っています (5-foldでのStratifiedKFoldで交差検証)。なお、`Lightning_T`関してその形状ゆえにDetectionがうまくいかないことが多かったため、`Lightning_T`のコネクタ全体を物体の領域としてアノテーションした場合 (`Lightning_T_all`) と、`Lightning_T`の先端の端子部分のみを物体の領域としてアノテーションした場合 (`Lightning_T_edge`) の2パターンでモデルを作成しました。学習時の設定は以下のとおりです。

```bash
--img 640 \
--batch 16 \
--epochs 300 \
--label-smoothing 0.1 \
--weights yolov5x.pt
```

### YOLOv5の推論
推論時の設定は以下のとおりです。推論時のポイントは以下の3点です。
- `Lightning_T_all`と`Lightning_T_edge`の場合の各Foldの合計10個のモデルを利用
- 学習時の入力サイズ (640) よりも大きい (832) 画像を入力として利用した上で、Test Time Augmentationを実施 (詳細は[こちら](https://github.com/ultralytics/yolov5/issues/303))
- 後処理で使うために、confidence scoreの大きい順にTop-3を保存しておく (`--max-det 3`)

```bash
--weights \
path/to/Lightning_T_all/cv0/weights/best.pt \
path/to/Lightning_T_all/cv1/weights/best.pt \
path/to/Lightning_T_all/cv2/weights/best.pt \
path/to/Lightning_T_all/cv3/weights/best.pt \
path/to/Lightning_T_all/cv4/weights/best.pt \
path/to/Lightning_T_edge/cv0/weights/best.pt \
path/to/Lightning_T_edge/cv1/weights/best.pt \
path/to/Lightning_T_edge/cv2/weights/best.pt \
path/to/Lightning_T_edge/cv3/weights/best.pt \
path/to/Lightning_T_edge/cv4/weights/best.pt \
--img 832 \
--augment \
--conf-thres 0.05 \
--max-det 3 \
--save-txt \
--save-conf \
--save-crop \
--half
```

### 後処理
上記の方法で推論を行った後にconfidence scoreが最大の結果を利用することでかなり正確にケーブルコネクタ部分の検出が可能でしたが、より正確に検出を行うために以下の後処理をしました。  

- confidence scoreのTop-1からTop-3に、以下のように検出結果が画像の中央であるか (画像の端ではないか) を判定
  1. Top-1が画像中央であれば、Top-1の検出結果を利用
  2. Top-1が画像中央でなくTop-2が画像中央であれば、Top-2の検出結果を利用
  3. Top-1、Top-2が画像中央でなくTop-3が画像中央であれば、Top-3の検出結果を利用
  4. Top-1、Top2、Top-3の全てが画像中央でなければT、Top-1の検出結果を利用
- ※ 検出結果の中心座標を(x_center, y_center)としたとき、`0.175 < x_center, y_center < 0.825`であれば中央であるとして判定

## Stage-2: クロッピングした画像で分類
### 学習
ターゲットのクラスが均等になるように、5-foldでのStratifiedKFoldで交差検証しました。  
モデルはEfficientNet、ResNetなどを使っています。学習の設定は以下のとおりです。

#### Data Augmentation

```python
import albumentations as A
additional_items = (
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Blur(blur_limit=(3, 13), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(sat_shift_limit=15, hue_shift_limit=10, p=0.5),
        A.Resize(self.size[0], self.size[1]),
    ]
)
transform = A.Compose(
    [
        *additional_items,
        A.Normalize(always_apply=True),
        ToTensorV2(always_apply=True),
    ]
)
```

#### Test Time Augmentation
入力サイズは学習時の1.3倍を利用

```python
import ttach as tta
transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Scale(scales=[1, 0.83, 0.67]),
    ]
)
```

### 結果

| model | input size | CV | Public | Private (コンペ終了後に確認) |
| -- | -- | -- | -- | -- |
| efficientnet_b4 | 380| 0.995019 | 0.981198 | 0.985915 |
| efficientnetv2_rw_m | 416 | 0.994271 | - | - |
| resnet18d | 224 | 0.991283 | - | - |
| resnet34d | 224 | 0.990037 | - | - |
| resnet50d | 224 | 0.994271 | 0.992279 | - | - |
| resnet101d | 256 | 0.993773 | 0.994022 | - | - |
| averaging-1 | - | 0.996513 | 0.983548 | 0.986656 |
| averaging-2 | - | 0.996513 | 0.984723 | 0.986656 |
| ensemble-lgbm | - | 0.994770 | 0.982373 | 0.985915 |

`averaging-1`: 以下の6つのモデルの予測確率をAveraging
- `efficientnet_b4`, `efficientnet_b4 + TTA`, `efficientnetv2_rw_m`, `resnet50d`, `resnet101d`, `resnet101d + TTA`

`ensemble-lgbm`: 以下の4つのモデルの予測確率をLightGBMでスタッキング
- `efficientnet_b4`, `efficientnetv2_rw_m`, `resnet50d`, `resnet101d`

`averaging-2`: 以下の7つのモデルの予測確率をAveraging
- `efficientnet_b4`, `efficientnet_b4 + TTA`, `efficientnetv2_rw_m`, `resnet50d`, `resnet101d + TTA`, `averaging-1`, `ensemble-lgbm`

### Pseudo-labeling
Stage-2の`averaging-2`の結果を使い、テストデータに疑似ラベルを付与しました。  
閾値は0.9として、予測確率が閾値を超えた1511サンプルに予測クラスを疑似ラベルとして付与しています。

## Stage-3: クロッピングした画像で分類 (疑似ラベルを追加)
### 学習
Stage-2と同じように学習しています。

#### Data Augmentation
Stage-2と同じ

#### Test Time Augmentation
Stage-2であまり効果がなかっったため、使わず。

### 結果

| model | input size | CV | Public | Private |
| -- | -- | -- | -- | -- |
| efficientnet_b4 | 380 | 0.997104 | 0.983548 | 0.985915 |
| efficientnet_b5 | 456 | 0.996019 | 0.985898 | 0.986656 |
| efficientnet_b6 | 528 | 0.995838 | 0.984723 | 0.984432 |
| efficientnet_b7 | 600 | 0.993305 | 0.983548 | 0.983691 |
| efficientnetv2_rw_m | 416 | 0.994752 | - | - |
| resnet50d | 224 | 0.995657 | - | - |
| resnet101d | 256 | 0.995657 | - | - |
| vit_base_patch16_224 | 224 | 0.995838 | - | - |
| gmlp_s16_224 | 224 | 0.992942 | - | - |
| nfnet_l0 | 224 | 0.996742 | - | - |
| averaging-1 | - | 0.996743 | **0.985898** | **0.986656** |
| averaging-2 | - | 0.997286 | - | - |
| averaging-3 | - | 0.997647 | **0.987074** | **0.986656** |
| ensemble-wgt-opt | - | 0.996562 | - | - |
| ensemble-lgbm | - | 0.997286 | - | - |
| ensemble-1D-CNN | - | 0.997647 | - | - |
| ensemble-2D-CNN | - | 0.997467 | - | - |

`averaging-1` (最終サブミッションの1つ目)
- `efficientnet_b5`を3パターンのseedでrandom seed averaging

`averaging-2`: 以下の6つのモデルの予測確率をAveraging
- `efficientnet_b4`, `averaging-1`, `efficientnet_b6`, `efficientnet_b7`, `resnet50d`, `resnet101d`

`ensemble-wgt-opt`: 以下の7つのモデルの予測確率をWeight OptimizationでWeighted Averaging
- `efficientnet_b4`, `efficientnet_b5` x 3 seed, `efficientnet_b6`, `resnet50d`, `resnet101d`

`ensemble-lgbm`: 以下の4つのモデルの予測確率をLightGBMでスタッキング
- `efficientnet_b4`, `efficientnet_b5` x 2 seed, `efficientnet_b6`, `efficientnetv2_rw_m`, `resnet50d`, `vit_base_patch16_224`, `nfnet_l0`, 

`ensemble-1D-CNN`: 以下の5つのモデルの予測確率を[1D-CNN](https://tawara.hatenablog.com/entry/2020/12/16/132415)でスタッキング
- `efficientnet_b4`,  `efficientnet_b5` x 2 seed, `efficientnet_b6`, `resnet50d`

`ensemble-2D-CNN`: 以下の5つのモデルの予測確率を[2D-CNN](https://tawara.hatenablog.com/entry/2020/12/16/132415)でスタッキング
- `efficientnet_b4`,  `efficientnet_b5` x 2 seed, `efficientnet_b6`, `resnet50d`

`averaging-3`: 以下のモデルの予測確率をAveraging (最終サブミッションの2つ目)
- `ensemble-1D-CNN` x 3 seed, `ensemble-2D-CNN` x 3 seed, `averaging-1`, `averaging-2`, `ensemble-wgt-opt`, `ensemble-lgbm` x 2 seed

# 余談
## コンペ序盤
コンペ序盤はなるべく手間をかけずに取り組もうと思っていました。`train`データセットのEDAから対象物体が画像中央にある傾向があることがわかったため、単純に全画像の中央60%部分をクロッピングして、そのまま`efficientnet_b4`を使い分類していました。  
スコアは、CV: 0.964917, Public: 0.968272 で、暫定ランキングでも上位に入っていました。 (コンペ終了後確認したところ、Private: 0.973313 で最終10位相当でした)

## コンペ中盤
全画像同一の大きさでクロッピングした場合の性能に限界を感じたため、YOLOv5を使って画像ごとにケーブルコネクタの箇所をクロッピングする方法に方針転換しました。  
初めは`train`データセットだけでYOLOv5を学習していましたがうまく検出できない画像が何枚かあり、`train`データセットのラベルの修正や、`additional`データセットのラベルの追加を行いました。また、エラー分析を通して学習データ中のラベルの間違いを多々発見したっため、ケーブルの種類のラベルの修正も行っていました。手作業でのラベル付けはなかなか大変で、コンペ中で一番時間を使ってしまいました...  
この段階で提出していたスコアは、CV: 0.995019, Public: 0.981198 で、暫定ランキングでも上位に入っていたと思います。(コンペ終了後確認したところ、Private: 0.985915 でこの時点で既に1位相当でした)

## コンペ終盤
Publicリーダーボードで順位が低下してきたため、様々な分類モデルを試してみたり、疑似ラベル付けをしたり、色々なアンサンブル手法を試していました。  
最終的には、CV: 0.997647、Public: 0.987074 まで向上し、暫定ランキングで2位まで上がりました。(Private: 0.986656 だったためPrivateスコアは efficientnet_b5 だけを使った場合と変わらず、アンサンブルは特に効果がなかったようです。)

## 実験環境
今回のコンペを通して、モデルの学習は以下の環境で行っていました。
- GPU: NVIDIA Tesla V100 x 1 (メモリ: 16 GB)  

efficientnet_b5でバッチサイズ: 8、efficientnet_b6でバッチサイズ: 6、efficientnet_b7でバッチサイズ: 3 という状態で学習させていましたが、(学習時間がかかるものの) 以外とうまく学習は進みました。  
efficientnet_b5では46 epoch学習を回して、2時間半くらいでした。(LinearWarmup (2 epochs) + CosineAnnealing (半周期 4epoch x 2 x 5.5サイクル) で合計46 epoch)