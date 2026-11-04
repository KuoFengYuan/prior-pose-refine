# prior-pose-refine

[English](README.md) | **繁體中文**

用 COLMAP 修正近似的相機姿態，同時校正焦距和鏡頭畸變，產生可以直接拿去做 3D Gaussian Splatting(3DGS)訓練的資料集。姿態和 3D 點可以來自任何能同時提供這兩者的來源:ARKit 加 LiDAR、LiDAR-慣性或視覺 SLAM、動作捕捉系統等等。

追蹤得到的姿態大致正確但不精確，而且像 ARKit 這類追蹤系統把相機當成沒有畸變的針孔相機，直接拿來訓練會有重影、細節發糊。prior-pose-refine 以先驗姿態為起點，用 3D 點挑選真正有重疊的影像來匹配，再用能描述手機鏡頭畸變的相機模型，把姿態和相機一起做 bundle adjustment,最後對齊回先驗模型的公尺座標系。這個工具是在 iPhone ARKit 掃描上開發和測試的，見[姿態和點雲可以從哪裡來](#31-姿態和點雲可以從哪裡來)。

## 結果

一份室內辦公室的掃描(iPhone ARKit 姿態加 Livox Mid-360 點雲，729 張有姿態的影像),用 LichtFeld Studio 訓練 30,000 iterations、高斯數上限 100 萬，每 8 張取 1 張當測試影像:

| 訓練用的姿態 | COLMAP 時間 | 測試 PSNR |
|---|---|---|
| 擷取 App 匯出的 ARKit 姿態 | — | 23.46 dB |
| 純 COLMAP(vocab tree + global_mapper) | 7:36 | 28.23 dB |
| prior-pose-refine,`--camera-model OPENCV` | 3:03 | 28.41 dB |
| **prior-pose-refine(預設)** | **3:30** | **30.84 dB** |

第二份掃描(1,505 張，iPhone 內建 LiDAR)的結果一致:30.55 dB,純 COLMAP 是 28.28 dB,匯出的 ARKit 姿態是 23.14 dB;COLMAP 時間 7:30 對 17:47。掃描怎麼拍攝、怎麼評估，見 [docs/benchmark.zh-TW.md](docs/benchmark.zh-TW.md#關於這些數據)。

提升主要來自相機模型：同一個流程下，預設的 RAD_TAN_THIN_PRISM_FISHEYE 比 OPENCV 高 2.4 dB。時間是在 Core Ultra 7 265K 加 RTX 5070(12 GB)、GPU 閒置時量測。完整數據和對照圖見 [docs/benchmark.zh-TW.md](docs/benchmark.zh-TW.md)。

## 目錄

1. [需求](#1-需求)
2. [安裝設定](#2-安裝設定)
3. [輸入](#3-輸入)
4. [快速開始](#4-快速開始)
5. [輸出](#5-輸出)
6. [訓練](#6-訓練)
7. [選項](#7-選項)
8. [怎麼判斷結果好不好](#8-怎麼判斷結果好不好)
9. [什麼時候改用 `--mode global`](#9-什麼時候改用---mode-global)
10. [疑難排解](#10-疑難排解)
11. [運作方式](#11-運作方式)
12. [檔案說明](#12-檔案說明)
13. [授權](#13-授權)

## 1. 需求

| 項目 | 說明 |
|---|---|
| 作業系統 | Linux,bash 5 以上 |
| COLMAP | 4.x,需要支援 CUDA(測試版本為 4.3.0.dev),要在 `PATH` 上 |
| Python | Python 3,需要 numpy 和 scipy |
| GPU | NVIDIA GPU。SIFT 擷取、匹配和 bundle adjustment 都會用到 |
| 訓練 | 任何能讀 COLMAP 資料集的 3DGS 訓練程式；測試用的是 LichtFeld Studio |

## 2. 安裝設定

```bash
git clone https://github.com/KuoFengYuan/prior-pose-refine.git
cd prior-pose-refine
pip install -r requirements.txt     # 或把 PYTHON 指向已經有 numpy 和 scipy 的 Python
cp local.env.example local.env      # 再填入這台機器的路徑
source local.env                    # 每開一個新的 shell 都要執行
```

每台機器的路徑不同，所以都寫在 `local.env`,這個檔案不納入版控。`prior-pose-refine` 只需要前兩個變數:

| 變數 | 意思 |
|---|---|
| `PATH` | 要包含 `colmap` 所在的目錄 |
| `PYTHON` | 要用的 Python(預設 `python3`);`--python` 可以覆蓋它 |
| `LICHTFELD`、`SCAN_ROOT`、`EXP_ROOT` | 選用，分別指向訓練程式、你的掃描和你的實驗資料夾 |

## 3. 輸入

輸入 `SCAN_DIR` 是一個資料夾，裡面有影像，以及一個存放先驗姿態、內參和點雲的 COLMAP 模型:

```
SCAN_DIR/
├── images/              影像，例如 frame_00001.jpg …
└── sparse/0/            COLMAP binary 格式
    ├── images.bin       每張影像的先驗姿態(world-to-camera,COLMAP 慣例)
    ├── cameras.bin      先驗內參(每張一組 PINHOLE 也可以)
    └── points3D.bin     3D 點(例如 LiDAR),和姿態在同一個座標系
```

需要符合這些條件:

- **姿態和點雲的單位是公尺。** 有幾個門檻是以公尺為單位(挑選匹配對、對齊的離群門檻、報告的標記),所以其他單位的模型，例如沒有尺度的 SfM 重建，不能直接使用。
- **姿態採用 COLMAP 慣例**:world-to-camera,相機的 x 軸向右、y 軸向下、z 軸向前。很多追蹤系統(包括 ARKit)輸出的是 camera-to-world、y 軸向上、z 軸向後的姿態，寫入 `images.bin` 前要先轉換。
- **姿態要大致正確。** 預設的 `fast` 模式做的是局部修正；在測試掃描上，它修正了最多 25 cm 的 ARKit 漂移。誤差更大時，改用 `--mode global`(第 9 節)。
- **只處理 `images.bin` 裡有的影像。** 沒有姿態的影像會被忽略。
- **所有影像是同一支相機拍的。** 工具會擬合一組共用的相機(預設 RAD_TAN_THIN_PRISM_FISHEYE),初始內參取先驗內參的中位數。
- **先驗內參對應的是 `cameras.bin` 裡記錄的影像解析度。** 如果 `--images` 指定的影像大小不同(例如縮過圖),工具會自動按比例縮放內參。
- **檔名要依拍攝順序排列。** 每張影像會額外和檔名順序前後各 3 張配對。
- **`points3D.bin` 要涵蓋相機看到的範圍。** 挑選匹配對和訓練的初始點雲都會用到這些點。沒有點雲的話，請加 `--matcher sequential`,初始點雲會自動只用 SfM 點。

這份文件和報告把這個模型裡的姿態、內參和點雲稱為「先驗」(prior)。

### 3.1 姿態和點雲可以從哪裡來

| 來源 | 狀態 |
|---|---|
| ARKit 姿態加 iPhone LiDAR 點雲 | 已測試 |
| ARKit 姿態加外接 LiDAR(Livox Mid-360)點雲 | 已測試 |
| 已經在手機上優化過的姿態(擷取 App 自己的優化) | 已測試 |
| LiDAR-慣性里程計(例如 FAST-LIO2)、視覺或視覺-慣性 SLAM、動作捕捉、機器人里程計、RTK-GNSS 加另外的點雲 | 符合上述條件就應該可以使用；尚未測試 |

[Splatify-Anything](https://github.com/KuoFengYuan/splatify-anything) 匯出的掃描已經是這個格式。其他來源要把姿態、內參和點雲寫成 COLMAP 模型，例如先寫成 COLMAP 的文字格式，再用 `colmap model_converter --output_type BIN` 轉換。

## 4. 快速開始

```bash
source local.env
./prior-pose-refine path/to/scan path/to/scan_opt
```

執行時會逐步印出每個階段和耗時:

```
prior-pose-refine: mode=fast matcher=covis rounds=3 camera=RAD_TAN_THIN_PRISM_FISHEYE init=merge
  scan:   …/scan
  prior:  …/scan/sparse/0
  images: …/scan/images
  out:    …/scan_opt
list_images           0.1 s
export_centers        0.1 s
feature_extraction    8.7 s
select_pairs          1.7 s
matching              13.8 s
init_model            0.2 s
triangulate_r0        11.4 s
bundle_adjust_r0      10.0 s
…
align                 0.4 s
restore_prior_poses   0.1 s
undistort             5.5 s
merge_points          0.3 s
== prior-pose-refine report
…
dataset ready: …/scan_opt/dataset
```

- 在 RTX 5070 上,390 張影像的掃描花了 1:46,805 張花了 3:11。
- 輸出資料夾已經存在時會停下來，不會覆蓋，要覆蓋請加 `--force`。
- 寫入前會檢查選項數值並解析符號連結。`OUT_DIR` 必須和 `SCAN_DIR`、影像資料夾、先驗模型分開；同一路徑、上層與下層路徑都會被拒絕，加 `--force` 也一樣。輸出資料夾包含主程式所在目錄時也會被拒絕。
- 每次輸出 1.2 到 2 GB,執行前先確認磁碟空間。

## 5. 輸出

```
OUT_DIR/
├── dataset/                 ← 訓練用這個
│   ├── images/              去畸變後的影像(PINHOLE)
│   └── sparse/0/            cameras.bin、images.bin、points3D.bin(SfM 點 + 先驗點雲)
├── model/                   去畸變前、已對齊先驗模型座標的模型(含畸變參數)
├── report.txt               結果摘要，見第 8 節
├── timing.tsv               每個階段的耗時
└── work/                    中間檔，可以刪掉
    ├── database.db          COLMAP 資料庫(特徵、匹配)
    ├── image_list.txt       有先驗姿態的影像清單
    ├── prior_centers.txt    先驗相機中心，給 model_aligner 用
    ├── pairs.txt            挑出來的匹配對
    ├── init/                先驗姿態的初始模型
    ├── tri*/ ba*/           每一輪三角化和 BA 的結果
    └── logs/                每個階段的完整 log
```

- `dataset/` 的座標系就是先驗模型的座標系，單位是公尺，可以和先驗點雲、姿態直接疊在一起。
- `model/` 保留了原始解析度和畸變參數。之後如果要用不同的方式去畸變，或輸出給其他軟體，用這份。不是每套軟體都支援 RAD_TAN_THIN_PRISM_FISHEYE;如果一定要 OPENCV 相機(例如要用含畸變的影像直接訓練),可以加 `--camera-model OPENCV` 重跑，但品質會少約 2.4 dB。
- `work/` 約 0.6 GB,確認結果沒問題後可以刪除。

## 6. 訓練

`dataset/` 是標準的 COLMAP 資料集，相機是 PINHOLE。用 LichtFeld Studio 訓練:

```bash
"$LICHTFELD" -d OUT_DIR/dataset --max-cap 1000000
```

- `--max-cap 1000000` 把高斯數上限設為 100 萬，上面所有結果都是用這個設定；可以依顯示卡記憶體調整(LichtFeld 預設是 500 萬)。
- **初始點雲**:`dataset/sparse/0/points3D.bin` 是 SfM 點加上先驗點雲(用相機中心算出的相似變換轉到模型座標)。兩份掃描實測，和只用 SfM 點相比,PSNR 持平(−0.03 / +0.02 dB),LPIPS 都變好，房間內懸空的 floater 少 41% / 67%。只用 LiDAR 點則會少約 1.4 dB。要只用 SfM 點，加 `--init-points sfm`。

要用測試影像評估品質時:

```bash
"$LICHTFELD" --headless -d OUT_DIR/dataset -o OUT_DIR/train \
    --max-cap 1000000 --eval --test-every 8
```

結果會寫在 `OUT_DIR/train/metrics_report.txt`,每張測試影像的 GT 和 render 對照圖在 `eval_step_*/`。

## 7. 選項

```
prior-pose-refine SCAN_DIR OUT_DIR [options]
```

| 選項 | 預設 | 說明 |
|---|---|---|
| `--images DIR` | `SCAN_DIR/images` | 影像資料夾 |
| `--prior-model DIR` | `SCAN_DIR/sparse/0` | 先驗 COLMAP 模型(姿態、內參、點雲) |
| `--mode fast\|global` | `fast` | `fast`:從先驗姿態出發做 BA(推薦)。`global`:COLMAP global_mapper 加重力方向，再對齊，見第 9 節 |
| `--rounds N` | `3` | 三角化加 BA 的輪數(只用於 fast)。2 輪快約 60 秒，但少約 0.2 dB;第 4 輪沒有幫助 |
| `--camera-model M` | fast:`RAD_TAN_THIN_PRISM_FISHEYE`<br>global:`OPENCV` | BA 和去畸變用的相機模型。`OPENCV_FISHEYE` 參數比較少、比較保守(少約 0.08 dB);`OPENCV` 少約 2.4 dB。`--mode global` 只支援 `OPENCV` |
| `--ba-iterations N` | `15` | 每輪 bundle_adjuster 的迭代上限，15 次以後結果已不再變化 |
| `--init-points sfm\|merge` | `merge`(先驗模型沒有 `points3D.bin` 時為 `sfm`) | 訓練資料集的初始點雲。`merge` = SfM 點加上先驗模型的點；`sfm` = 只用 SfM 點。見第 6 節 |
| `--matcher covis\|sequential` | `covis` | `covis`:用先驗姿態和點雲共視挑選匹配對。`sequential`:COLMAP 依拍攝順序匹配，沒有點雲時使用 |
| `--top-k N` | `25` | 每張影像配對的共視影像數 |
| `--seq N` | `3` | 每張影像額外配對的前後相鄰張數 |
| `--overlap N` | `20` | `sequential` 模式下，前後各匹配幾張 |
| `--gpu N` | `0` | 使用的 GPU 編號 |
| `--gpu-workers N` | `3` | 同一張 GPU 上同時跑幾個 SIFT 擷取 |
| `--max-features N` | `4096` | 每張影像的 SIFT 特徵數 |
| `--align-max-error M` | `0.1` | 對齊先驗姿態時的離群門檻(公尺) |
| `--min-obs N` | `20` | 觀測數少於 N 的影像沿用先驗姿態，見[運作方式](docs/method.zh-TW.md#步驟-8-觀測太少的影像)。設成 0 就關閉 |
| `--python PATH` | `$PYTHON` 或 `python3` | 有 numpy 和 scipy 的 Python |
| `--force` | | 覆蓋已存在的輸出資料夾；與輸入重疊的路徑一律拒絕 |

這些預設值是實驗找出來的最佳設定，除非遇到問題，否則用預設值就好。

`--rounds`、`--ba-iterations`、`--overlap`、`--gpu-workers` 和 `--max-features` 必須是正整數。`--gpu`、`--top-k`、`--seq` 和 `--min-obs` 可以是 0；所有整數選項的上限都是 2147483647。使用 `--matcher covis` 時，`--top-k` 和 `--seq` 不能同時是 0。`--align-max-error` 必須是大於 0 的有限數值。缺少選項值或數值超出範圍時，會在移除既有輸出之前停止。

## 8. 怎麼判斷結果好不好

看 `OUT_DIR/report.txt`。以下是一份 390 張影像的掃描於 2026-11-04 重跑的報告，可以當作「正常」的參考值:

```
Registered images: 390            ← 應該等於先驗模型的影像數
Points: 120528
Mean reprojection error: 0.690695px
refined camera (with distortion):
camera 1: RAD_TAN_THIN_PRISM_FISHEYE 1920x1440 1363.97 1363.82 963.985 724.283 0.622796 -0.103451 -8.14841 37.5694 -62.5652 36.0914 …
agreement with the prior poses:
390/390 prior frames in the model | distance to the prior camera centers: median 1.8 cm, max 3.6 cm | after a similarity
(scale 1.0000, rotation 0.00 deg): median 1.8 cm, p95 3.3 cm
frames with fewer than 20 observations (prior pose kept): 0
frames more than 10 cm from their prior pose after the similarity: 0
```

這次在 RTX 5070 上重跑花了 103.4 秒，從 GPU 閒置時開始並記錄使用率。這些是參考值，不要求逐字一致：GPU 非確定性可能讓點數相差數百點、重投影誤差相差約 0.001 px。輸入檢查不代表畫質驗證；PSNR 是否改善仍須透過 3DGS 評估。

| 項目 | 正常 | 有問題時 |
|---|---|---|
| Registered images | 等於先驗模型的影像數 | 少很多：有些影像沒有足夠的匹配，檢查 `work/logs/matching.log` |
| Mean reprojection error | 約 0.7–0.8 px(`--camera-model OPENCV` 約 1.15 px) | 1.5 px 以上：姿態或內參沒有收斂，可以試 `--rounds 4`、`--camera-model OPENCV_FISHEYE` 或 `--mode global` |
| fx | 和先驗的 fx 差 1% 以內(ARKit 的情況) | 差很多：相機模型可能不適合(例如超廣角鏡頭),或先驗內參有誤 |
| 畸變係數 | 高階係數很大、正負交錯是正常的，它們在影像範圍內互相抵消 | 不用看係數，看重投影誤差 |
| 對齊後和先驗姿態的差距 | 中位數 1–2 cm(ARKit 的情況) | 10 cm 以上：追蹤可能有問題，改用 `--mode global` |
| `frames with fewer than 20 observations` | 0 | 列出的影像沒有足夠的特徵(通常是模糊、白牆),姿態沿用先驗姿態。少數幾張沒關係；很多張的話，代表那段拍攝的品質不好 |
| `frames more than 10 cm from their prior pose` | 0 或少數 | 列出的影像被修正超過 10 cm,括號裡是觀測數。觀測數多(數百以上)通常是修正了先驗姿態的漂移；觀測數少的話，要實際看那幾張影像 |

## 9. 什麼時候改用 `--mode global`

預設的 `fast` 模式假設先驗姿態大致正確，只需要局部修正。以下情況請改用 `--mode global`:

- 拍攝中途追蹤跳動、重新定位，或大範圍漂移。
- 報告裡和先驗姿態的差距中位數超過 10 cm,或重投影誤差偏高。

`global` 模式不使用先驗姿態本身，只用兩樣東西:

1. 重力方向，用來輔助 global_mapper 的旋轉估計。程式會從相機的「影像朝下」方向推算，相機大致保持直立時(手持或地面機器人)是正確的，但例如朝下拍的空拍機相機就不適用。
2. 先驗相機中心，用來在最後把模型對齊回公尺座標系。

`global` 模式只支援 OPENCV 相機模型。在測試掃描上，品質和純 COLMAP 一樣(28.26 dB),比 `fast` 模式低約 2.6 dB,時間也比較長(global_mapper 本身約 245 秒)。

```bash
./prior-pose-refine SCAN_DIR OUT_DIR --mode global
```

## 10. 疑難排解

| 訊息或狀況 | 處理方式 |
|---|---|
| `lacks numpy/scipy` | 在 `local.env` 設定 `PYTHON`,或加 `--python PATH`,或 `pip install -r requirements.txt` |
| `colmap is not on PATH` | 在 `local.env` 把 COLMAP 所在目錄加進 `PATH`,再 `source local.env` |
| `OUT_DIR exists` | 換一個輸出資料夾，或加 `--force` |
| `OUT_DIR overlaps an input` 或 `OUT_DIR contains the pipeline directory` | 改用獨立的輸出資料夾，慣例是 `<SCAN_DIR>_opt`；檢查會解析符號連結，`--force` 也不能略過 |
| `requires a value`、`must be an integer` 或 `must be a finite number` | 補上選項值，並遵守第 7 節的範圍；`--min-obs 0` 仍可使用 |
| `--top-k and --seq cannot both be zero` | 啟用共視配對或相鄰影像配對，或改用 `--matcher sequential` |
| `select-pairs: only N points` | 掃描沒有 LiDAR 點雲，加 `--matcher sequential` |
| `no prior points are visible` 或 `no matching pairs were selected` | 檢查先驗座標慣例、公尺單位和點雲覆蓋範圍；必要時增加 `--seq`，或使用 `--matcher sequential` |
| `camera centers are coincident or collinear` | 這些相機中心無法決定相似變換的旋轉；請使用相機至少沿兩個獨立方向移動的掃描。平面上的拍攝路徑可以使用 |
| `world down` 的分量無效或沒有穩定方向 | 檢查先驗旋轉；直接呼叫 `write-pose-priors` 時，可指定有限且非零的 `--world-down x,y,z` |
| `prior frames are not in the database` | `--images` 資料夾裡缺少某些影像，或檔名和 `images.bin` 對不起來 |
| `--mode global supports only --camera-model OPENCV` | `global` 模式不要指定其他相機模型 |
| 報告的 `Points: 0` | 三角化沒有產生任何點。如果是自己改過流程：擷取和匹配時相機必須是 OPENCV(否則幾何驗證會全部失敗);point_triangulator 必須加 `--Mapper.max_extra_param 1e9`(否則高階畸變係數會被當成異常值) |
| 某個階段 `FAILED` | 程式會印出該階段 log 的最後 20 行，完整 log 在 `OUT_DIR/work/logs/<階段>.log` |
| GPU 記憶體不足 | 減少 `--gpu-workers`,例如改成 1 |
| 磁碟空間不足 | 把 `OUT_DIR` 放在空間足夠的磁碟；確認結果後刪除 `work/` |

## 11. 運作方式

簡單來說:SIFT 特徵 → 用先驗姿態和點雲共視挑選匹配對 → 從先驗姿態出發，用 RAD_TAN_THIN_PRISM_FISHEYE 相機做 3 輪三角化和 bundle adjustment → 對齊回先驗模型的公尺座標系 → 去畸變。每個步驟和背後的理由見 [docs/method.zh-TW.md](docs/method.zh-TW.md)。

## 12. 檔案說明

| 檔案 | 說明 |
|---|---|
| `prior-pose-refine` | 主程式，串起整個流程 |
| `prior_pose_refine.py` | 輔助子指令：讀取先驗模型、挑選匹配對、建立初始模型、寫入先驗、把觀測太少的影像還原成先驗姿態、檢查對齊、合併初始點雲。每個都可以單獨執行，用 `--help` 查看說明 |
| `requirements.txt` | Python 套件需求 |
| `tests/test_prior_pose_refine.py` | 輸入保護、數值範圍、幾何、配對選擇及 COLMAP 姿態一致性的回歸測試 |
| `.github/workflows/checks.yml` | Python 3.11、3.12、3.13 的 GitHub Actions 檢查，包含 pycolmap 往返測試 |
| `local.env.example` | 本機路徑設定的範本，複製成 `local.env`(不納入版控)後填入 |
| `docs/method.zh-TW.md` | 流程怎麼運作、為什麼這樣設計 |
| `docs/benchmark.zh-TW.md` | 完整實驗數據和對照圖 |
| `AGENTS.zh-TW.md` | 貢獻者工作守則與交付流程 |
| `LICENSE` | MIT 授權 |

使用與流程相同的 Python 執行回歸測試：

```bash
source local.env
"$PYTHON" -I -B tests/test_prior_pose_refine.py -v
```

測試使用 numpy、scipy 和標準函式庫。COLMAP 二進位往返測試另需 pycolmap，未安裝時會略過該項；主流程不依賴 pycolmap。合成測試不能取代完整掃描重跑或 3DGS 畫質評估。

推送至 `main`、任務分支或提出 PR 時，GitHub Actions 會安裝 pycolmap 並執行回歸測試，檢查 CPU 行為與二進位模型相容性。完整 CUDA 流程與 3DGS 評估仍使用本機掃描，另外驗證。

## 13. 授權

MIT,見 [LICENSE](LICENSE)。
