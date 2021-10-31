import os
import pickle
from glob import glob

import albumentations as A
import cv2
import pandas as pd
from tqdm import tqdm

from src.data_vis import read_cv_image

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class PostProcessor:
    class2target = {
        "USB_Micro_B": 0,
        "USB_Micro_B_3.1": 1,
        "USB_Micro_B_W": 2,
        "USB_Type_A": 3,
        "USB_Type_B": 4,
        "USB_Type_C": 5,
        "USB_Mini": 6,
        "Lightning": 7,
        "Lightning_T": 8,
        "DisplayPort": 9,
        "Mini_DisplayPort": 10,
        "RJ_45": 11,
        "HDMI": 12,
        "VGA": 13,
        "Dock": 14,
    }

    def __init__(self, path: str = "path/to/dataset/dir") -> None:
        self.path_test_labels = os.path.join(CUR_DIR, "outputs/detect/test_1280/labels")
        self.path = path
        self.test_df = pd.read_csv(os.path.join(path, "sample_submission.csv"))[["filename", "class"]]
        self.test_df["x_center"] = -1.0
        self.test_df["y_center"] = -1.0
        self.test_df["width"] = -1.0
        self.test_df["height"] = -1.0

        # path to original images
        self.path_test_imgs = os.path.join(path, "test")

        # path to output images
        self.path_test_out = os.path.join(path, "preprocessed", "test_all_clean_cropped")

    def complement_labels(self) -> None:
        """
        ラベルがついてない場合、"-1,  0.5, 0.5, 0.6, 0.6, -1" で補完する
        """
        test_labels = glob(f"{self.path_test_labels}/*")
        test_labels = set([p.split("/")[-1] for p in test_labels])
        assert len(test_labels) == len(self.test_df), "Length does not match"

        for filename, target in zip(self.test_df["filename"], self.test_df["class"]):
            label_name = filename.replace(".jpg", ".txt")
            if label_name not in test_labels:
                print(f"{label_name=} not found, newly create...")
                with open(os.path.join(self.path_test_labels, label_name), "w") as f:
                    body = f"{self.__class__.class2target[target]} 0.5 0.5 0.6 0.6 -1 \n"
                    f.write(body)

    def post_process(self) -> None:
        """
        後処理をして、予測ラベルをtest.csvに出力する
        """
        for filename in tqdm(self.test_df.filename):
            # ラベルを読み込む
            path_label = os.path.join(self.path_test_labels, filename.replace(".jpg", ".txt"))
            with open(path_label, "r") as f:
                preds = f.readlines()
            assert 0 < len(preds) <= 3, "Preds length does not match"

            # ラベルの位置を判定
            valid = False
            for pred in preds[::-1]:
                target, x_center, y_center, width, height, conf = map(float, pred.rstrip().split())

                # 検出結果が画像の端の場合を処理
                if (0.175 < x_center < 0.825) and (0.175 < y_center < 0.825):
                    valid = True  # 有効な検出結果と判定
                    break
                else:
                    print(f"invalid detection: {filename=}")

            # 有効な検出結果が得られた場合
            if valid:
                pred_p = [target, x_center, y_center, width, height, conf]
            # 有効な検出結果が得られなかった場合、Top-1を利用する
            else:
                with open(path_label, "r") as f:
                    pred_p = list(map(float, f.readlines()[-1].rstrip().split()))

            # 後処理をした予測ラベルをtest.csvに出力する
            self.test_df.loc[self.test_df["filename"] == filename, "x_center"] = pred_p[1]
            self.test_df.loc[self.test_df["filename"] == filename, "y_center"] = pred_p[2]
            self.test_df.loc[self.test_df["filename"] == filename, "width"] = pred_p[3]
            self.test_df.loc[self.test_df["filename"] == filename, "height"] = pred_p[4]

        assert 1.0 >= self.test_df["x_center"].min() >= 0.0, "Value error"
        assert 1.0 >= self.test_df["y_center"].min() >= 0.0, "Value error"
        assert 1.0 >= self.test_df["width"].min() >= 0.0, "Value error"
        assert 1.0 >= self.test_df["height"].min() >= 0.0, "Value error"
        self.test_df.to_csv(os.path.join(self.path, "test.csv"))

    def crop_images(self) -> None:
        """
        YOLOv5の推論したラベルを利用して、画像をcroppingする
        - originalを正方形にpaddingする
        - labelを使ってcropping
        - paddingする
        - resizeせずに保存
        """
        for row in self.test_df.iterrows():
            row = row[1]
            # load label
            filename = row.filename
            x_center = row.x_center
            y_center = row.y_center
            width = row.width
            height = row.height

            # load original image
            img_path = os.path.join(self.path_test_imgs, filename)
            img = cv2.imread(img_path)

            # padding to square
            H, W, _ = img.shape
            print(f"{filename}: ({H}, {W}) -> ", end="")
            composer = A.Compose(
                [
                    A.PadIfNeeded(always_apply=True, min_height=max(H, W), min_width=max(H, W), border_mode=1),
                ]
            )
            img = composer(image=img)["image"]

            # cropping
            H, W, _ = img.shape
            x_center_pos = W * x_center
            y_center_pos = H * y_center

            x_min = x_center_pos - W * width / 2
            x_max = x_center_pos + W * width / 2
            y_min = y_center_pos - H * height / 2
            y_max = y_center_pos + H * height / 2
            composer = A.Compose(
                [
                    A.Crop(always_apply=True, x_min=int(x_min), y_min=int(y_min), x_max=int(x_max), y_max=int(y_max)),
                ]
            )
            img = composer(image=img)["image"]

            H, W, _ = img.shape
            print(f"({H}, {W})")

            # save image
            cv2.imwrite(os.path.join(self.path_test_out, filename), img)

    def padding_images(self, border_mode: int = 1) -> None:
        """
        croppingした画像を読み込んで、paddingして保存する
        """
        for filename in tqdm(self.test_df.filename):
            img = read_cv_image(os.path.join(self.path_test_out, filename))
            size = max(img.shape[0:2])
            composer = A.Compose(
                [
                    A.PadIfNeeded(always_apply=True, min_height=size, min_width=size, border_mode=border_mode),
                ]
            )
            img = composer(image=img)["image"]

            output_dir = os.path.join(self.path, "preprocessed", f"test_all_clean_cropped_pad{border_mode}")
            save_path = os.path.join(output_dir, filename.replace(".jpg", ".pickle"))
            with open(save_path, "wb") as f:
                pickle.dump(img, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    pp = PostProcessor(path=os.path.join(CUR_DIR, "input"))
    pp.complement_labels()
    pp.post_process()
    pp.crop_images()
    pp.padding_images(border_mode=1)


if __name__ == "__main__":
    main()
