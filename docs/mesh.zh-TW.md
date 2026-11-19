# LichtFeld Studio 分塊匯出 mesh

[English](mesh.md) | **繁體中文**

從已有的訓練結果匯出三角形 PLY，以及內嵌照片底色貼圖的 GLB。各區塊都使用公尺世界座標，可以單獨載入，也可以直接拼在一起。貼圖會保留拍攝當下的光照。反光材質在原始相機移動時可能呈現不同外觀。

## 安裝與使用

使用支援外掛 API v1 的 LichtFeld Studio 0.5.5，另備一個 Python 環境執行匯出。3DGS 與混合模式需要 CUDA、PyTorch 以及提供 `GaussianSplat3d.render_depths` 的 fVDB Reality Capture；依 CUDA 環境參照 [fVDB 安裝文件](https://fvdb-reality-capture.readthedocs.io/latest/installation.html)。只用感測深度可用 CPU 執行。

在 repository 中使用匯出用的 Python 執行：

```bash
python -m pip install -r plugins/tiled_mesh/requirements.txt
export MESH_PYTHON="$(command -v python)"
export LICHTFELD="$(command -v LichtFeld-Studio)"  # or source local.env
mkdir -p ~/.lichtfeld/plugins
cp -R plugins/tiled_mesh ~/.lichtfeld/plugins/  # first install
"$LICHTFELD" plugin check tiled_mesh
```

在 LichtFeld 外掛管理員載入外掛。打開 **Mesh Export**；若啟動程式時未設定 `MESH_PYTHON`，在 **Worker Python** 選擇執行檔。選擇深度來源與 voxel 尺寸，勾選 **Bake photo textures**（預設開啟）後按 **Export Saved Model**。啟用 **Export after training**，訓練成功、且已存檔的專案綁定最終迭代的 checkpoint 時，就會執行相同的匯出。請在 LichtFeld 開啟訓練結束時儲存專案。若已存檔的專案仍可找到訓練資料集，面板會自動帶入。資料集移動過或模型是獨立的 Gaussian PLY 時，請手動指定對應的資料集。每次執行會建立獨立目錄，內含 `export.log`、`mesh/mesh.json`、各區塊的 PLY 及可重用的深度圖。勾選貼圖後，可將 GLB 匯入 Blender，用「材質預覽」查看。面板與 `mesh/timings.json` 會列出深度準備、融合與烘焙各花了幾秒。

在面板中重建局部時，在 **Prepared views** 選取先前的 `views/`，使用相同的 voxel 尺寸與每塊格數，輸入 `mesh.json` 的三個索引後按 **Rebuild from Prepared Views**。索引留白會重建整個場景。

## 命令列

```bash
"$MESH_PYTHON" -I plugins/tiled_mesh/export_mesh.py export-mesh \
  --trained OUT_DIR/train --dataset OUT_DIR/dataset \
  --splat OUT_DIR/train/project.licht --scan SCAN_DIR \
  --source hybrid --voxel 0.02 --bake-textures --output OUT_DIR/mesh
```

- `--source splat` 使用訓練後的 Gaussian PLY 或已存檔的訓練專案，以及相符的最終相機。`.licht` 交由 LichtFeld 轉成 PLY。場景和相機須在同一個公尺座標系。若曾縮放、移動節點或合併多個模型，請提供座標一致的相機與 splat。
- `--source lidar` 只需 `--scan`，可略過 `--splat`。依影像名稱讀取光軸深度（float32）與信心值（uint8），依逐幀 RGB 內參和深度解析度換算光線，只採用信心等級 2。此格式假設深度已與 RGB 對位且共用光心。獨立深度相機須先校正並重投影。
- `--source hybrid` 同時需要兩者。優先採用實測深度，補洞圖會排除已有可靠實測值的像素，以及落在鄰近感測視角實測自由空間內的 3DGS 表面。
- `--cameras DIR` 可明確指定最終 COLMAP 模型。提供 `--trained` 時優先找 `cameras_refined/`，其次使用去畸變資料集的 `sparse/0`。需要公尺尺度的最終 PINHOLE 或 SIMPLE_PINHOLE 相機；含畸變的 `model/` 不適用。
- 預設 voxel 邊長 2 cm、每塊邊長 64 格、深度圖寬度上限 320 像素、最大光軸深度 20 m、每批最多渲染 25 萬個 splat。依掃描可靠距離設定 `--max-depth`、依影像細節設定 `--width`、依可用顯存設定 `--batch-splats`。voxel 邊長減半，每塊在三個方向的實際長度也會減半。
- `--exclude FILE` 每行填一個已註冊影像名稱。留出幀可保留感測深度供評估，其渲染深度與實測資料皆不參與融合或多視角檢查，照片也會排除於烘焙之外。

調整解析度、比較深度來源或重建單一區塊時，可以使用同一份深度快取。`--tile` 後的三個整數取自 `mesh.json` 的區塊索引；省略時會匯出整個場景。每次指定一個新目錄。

```bash
"$MESH_PYTHON" -I plugins/tiled_mesh/export_mesh.py build-mesh \
  --views OUT_DIR/mesh/views --source hybrid --voxel 0.04 \
  --tile 0 0 0 --output OUT_DIR/mesh-rebuilt-tile
```

也可以用相同的輸入選項執行 `prepare-views`，單獨建立快取。匯出中斷時會保留先前已發佈的結果。玻璃、細桿、只有單側視角的區域可能出現孔洞，若要做碰撞或量測，先檢查這些位置。

## 細節、貼圖與處理時間

可先使用 2 cm voxel 搭配原尺寸照片。需要更密的表面時可用 `--voxel 0.015 --width 512 --tile-cells 128`。多張影像都看得到的細節，才可能受益於較細的網格。增加深度圖寬度會拉長準備深度、融合的時間。照片貼圖可獨立維持原始解析度。

每塊預設最多選八張照片，依可見範圍、取樣密度、表面角度及深度一致性挑選。勾選 **Bake both sides**（`--two-sided-textures`）後，其餘三角形會翻轉法線再挑一次照片，並把允許的深度差縮小一半，適合薄片或面朝向不一致的區域。每一側最多取 `--bake-views` 張照片，雙面材質在同一三角形的兩側顯示相同顏色。可見區域加上邊緣像素後，打包成 PNG 內嵌在 GLB。缺乏可靠照片的表面保留灰色，`textures.json` 記錄有照片覆蓋的面積比例。不同視角的光照或曝光差異可能留下色差。

可用 `--texture-downsample 2` 快速預覽半解析度貼圖。`--atlas-size` 指定單張貼圖邊長上限（512、1024、2048 或 4096 像素，預設 2048），一個區塊可以有多張貼圖。`--bake-views` 設定每塊最多使用幾張照片。只要更換貼圖時，在面板選 **Existing mesh to rebake**，按 **Rebake Textures**，或執行下列指令。

```bash
"$MESH_PYTHON" -I plugins/tiled_mesh/export_mesh.py bake-textures \
  --mesh OUT_DIR/mesh --images OUT_DIR/dataset/images \
  --texture-downsample 2 --output OUT_DIR/mesh-preview
```

這會沿用現有的三角形和深度圖。深度快取若在另一個目錄，加上 `--views DIR`。移動過的資料集或舊版快取需指定對應的去畸變照片。加上 `--tile X Y Z` 可以只重烘一塊。命令列的 `export-mesh` 和 `build-mesh` 使用 `--bake-textures` 啟用貼圖。

## 驗收步驟

1. 選 **LiDAR**、4 cm voxel 匯出。在 Blender 或 CloudCompare 一次打開輸出資料夾的 PLY，檢查跨區塊的牆面能否在世界座標接起來。
2. 從 `mesh.json` 挑一塊，使用同一份深度快取、來源、voxel 尺寸與每塊格數重建。在同一個執行環境中，用 `cmp` 比較兩個 PLY 應逐位元相同。
3. 測試自動匯出時，先勾選 **Export after training** 並開啟訓練結束時儲存專案。訓練成功後檢查新的匯出 log 與 mesh 清單；使用者中止訓練時不會排入匯出。

要重現實測的留出方式，從 `per_image_metrics.json` 取出有 test 評估的影像檔名，每行一個，傳給 `prepare-views --exclude`。三種來源與兩種 voxel 尺寸共用這份快取。

## 大場景如何分塊

先依相機深度排序，把 splat 分批送入 GPU 渲染，再依剩餘透光率合成各批的深度和不透明度，最後除以總不透明度。只保留總不透明度至少 0.5、且另有兩個具位移視角支持的像素。準備深度圖時，模型參數與當幀排序會占用主記憶體。專案轉 PLY 的資源需求取決於 LichtFeld 的載入方式；已有 PLY 可以直接使用。

CPU 融合只為深度像素涵蓋的空間和四個 voxel 寬的截斷帶建立區塊索引。每次配置一塊網格，相鄰區塊共用邊界取樣點，每個 Marching Cubes 網格單元只屬於一塊。單元的八個角都至少有兩次觀測才產生三角形。融合時的網格記憶體取決於每塊的格數，影格索引隨可見範圍增加，深度快取與輸出 mesh 存在磁碟。評估時一起看留出深度誤差與覆蓋率，以分辨表面缺失和位置偏差。[Mesh 匯出實測](benchmark.zh-TW.md#mesh-匯出)
