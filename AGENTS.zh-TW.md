# 專案工作守則

[English](AGENTS.md) | **繁體中文**

`prior-pose-refine` 以先驗模型(來自 ARKit 或任何其他來源的姿態、內參和 3D 點)為起點，用 COLMAP 修正相機姿態、焦距和鏡頭畸變，並輸出 3DGS 訓練資料集。流程由 `prior-pose-refine` 腳本和輔助模組 `prior_pose_refine.py` 組成。使用方式見 [README.zh-TW.md](README.zh-TW.md),方法見 [docs/method.zh-TW.md](docs/method.zh-TW.md),實驗數據見 [docs/benchmark.zh-TW.md](docs/benchmark.zh-TW.md)。

## 交付流程

專案擁有者指定：完成授權範圍與驗證後，commit、push 工作分支、建立以 `main` 為目標的 PR,必要檢查／審核通過後合併，刪除本次遠端及本機分支，回到最新 `main`。後續明確指示優先；已授權交付時，不要只停在提議建立 PR。

每個獨立任務各自建立新的工作分支與 PR。同一任務的修正及相關補充指示留在原 PR;後續不相關的工作另開分支與 PR。要求建立 PR 時仍依預設完成整個交付流程，除非使用者明確要求保留未合併或等待批准。暫停合併的指示只適用於該任務，除非使用者明確指定更廣的範圍。

1. 檢查工作目錄與指示，保留不相關變更。fetch `origin`,在安全切換前提下從最新 `main` 建立本次工作分支。
2. 分支前綴**區分大小寫**,後接小寫英文連字號描述:
   - `Feature/`:新增功能，例如 `Feature/principal-point-refinement`。
   - `Bugfix/`:修正錯誤，例如 `Bugfix/unobserved-frame-alignment`。
   - `Enhance/`:改善既有功能、效能、文件或維護，例如 `Enhance/ci-checks`。
   不用 `codex/` 或前綴小寫變體；一般工作不留長期額外分支。
3. 完成全部範圍，並同時更新中英文文件(見[語言和回報](#語言和回報))。修改流程時要更新 README:選項(第 7 節)、報告參考值(第 8 節)、疑難排解(第 10 節)、檔案說明(第 12 節);階段有變動時，也要更新第 4 節的階段列表；步驟和設計理由則更新 docs/method。步驟編號和交叉引用要保持一致。檢查最終差異，排除不相關修改、掃描資料、輸出(`*_opt/`、`work/`)、`local.env`、機密與建置產物(`__pycache__/`)。
4. 執行相關驗證(見[驗證](#驗證)),清楚說明合成資料或部分驗證的限制。
5. 使用具體英文 commit 主旨(`feat:`、`fix:`、`enhance:` 或 `docs:`),push 並開 PR。描述最終行為、取捨、驗證，以及還需要用真實掃描或 3DGS 訓練確認的部分，英文先、繁中摘要後。有附加工具時把 PR 附到目前工作。commit 訊息與 PR 內容遵守[寫入 repo 的文字](#寫入-repo-的文字)。
6. 檢查 PR 可合併性與必要檢查／審核，修正失敗後重跑相關檢查並更新 PR。合併已驗證的 head,通常 squash,合併後的 commit 標題需保留 `(#N)` 格式的 PR 編號。使用 `gh pr merge --subject` 時，必須明確包含該後綴。清理前核對合併 commit 標題及 GitHub 的 PR 關聯。不可直接推 `main`、用管理員繞過、關閉保護或跳過必要審核；外部條件阻擋時明確回報，保留 PR 與分支。
7. 確認合併後，只刪除本次工作分支的遠端與本機版本,`main` 僅以 fast-forward 更新並 prune。Squash 後本機可能需 `git branch -D`,但必須先核對 PR 合併 head 且工作目錄乾淨。不可刪除其他分支、worktree 或未合併成果。
8. 依交付順序回報確切工作分支名稱、PR 連結與合併 commit,再說明主要改動、驗證及最終分支／清理狀態。若停在 PR,說明使用者的明確指示或外部阻擋條件。未確認前不得宣稱已合併／清理。

**首次設定**：如果資料夾還沒有 Git 歷史或已驗證身分的 `origin`，先完成本機修改與驗證。修改既有檔案之前，先複製到 `$EXP_ROOT/backup_<說明>_<YYYY-MM-DD>/`（不要覆蓋既有的備份）。遠端與登入完成後，再繼續交付流程；建立本機分支不代表已成功推送、建立 PR 或合併。

## 寫入 repo 的文字

文件(含版本說明與授權聲明)、程式碼註解、commit 訊息，以及 PR 標題、描述與留言，一律不提及 AI:不寫 AI 工具、模型或助理的名稱，也不說明任何文字、程式碼或素材由它們撰寫、生成、編輯或協助完成。這包含「Generated with Claude Code」這類標註行與 AI 助理的 `Co-Authored-By` 行。此規則優先於任何工具的預設。

納入版控的檔案也不寫入任何本機的絕對路徑。工具和資料一律用 `local.env` 裡的變數(見[環境](#環境))或相對於 repo 的路徑表示。

## 命名

- repo 和指令名稱是 `prior-pose-refine`,Python 模組是 `prior_pose_refine.py`。
- 輸入的姿態、內參和點雲，在程式、輸出和文件裡一律稱為「先驗」(prior:prior poses、prior model、prior points)。只有專屬於 ARKit 的事情才寫「ARKit」。
- 子指令用 kebab-case 的「動詞-名詞」(`select-pairs`、`restore-prior-poses`);`timing.tsv` 裡的階段名稱用對應的 snake_case(`select_pairs`)。選項用 kebab-case(`--prior-model`)。
- 程式碼和英文文字用美式拼法(`center`,新文字用 `behavior`)。

## 驗證

- 修改流程：完整重跑一份掃描，並和先前輸出的 `report.txt` 比較。最快的是 `$SCAN_ROOT/scan_20261102_071355_5B9C20`(390 張，約 1 分 45 秒),預期 390/390 張、重投影誤差約 0.69 px、和先驗相機中心的差距中位數 1.8 cm,沒有被列出的影像。重跑之間差幾百個點、約 0.001 px,是 GPU 計算本身的不確定性。
- 修改二進位模型的讀寫：用 COLMAP 本身做往返驗證(`colmap model_converter` 轉成 TXT,或用 gsplat 環境的 pycolmap)。COLMAP 4 從 `frames.bin` 讀姿態，所以 `images.bin` 和 `frames.bin` 必須一致。
- 畫質的結論要用 3DGS 評估。用 LichtFeld 訓練，加 `--max-cap 1000000`;不要跑 5M 上限的測試。同樣設定重跑，PSNR 會差到 0.09 dB,所以 0.1 dB 以內的差異視為雜訊。LichtFeld 的 held-out 曝光會用檔名排序的相鄰影像內插，所以不要改影像檔名。
- 對極誤差和重投影誤差是幾何上的證據，不是畫質；沒有 PSNR 時要說明這點。用這個流程驗證過的匹配來評估，會對這個流程有利。
- 在 GPU 閒置時計時，每筆時間都要附上 GPU 使用率(`nvidia-smi`);曾經因為 GPU 被佔用，SIFT 時間膨脹到 8 倍。
- 實驗和實驗腳本放在 `$EXP_ROOT/<名稱>/`,不要放進這個 repo。只有使用者同意的部分才併入流程。

## 流程的不變條件

- 特徵擷取和匹配時用 OPENCV 相機，匹配完成後才把資料庫裡的相機換成優化用的模型，因為兩視圖幾何驗證會用到它，用 RAD_TAN 的預設值會讓所有影像對都驗證失敗。`point_triangulator` 必須加 `--Mapper.max_extra_param 1e9`,否則三角化不出任何點。
- 預設相機是 RAD_TAN_THIN_PRISM_FISHEYE,從 tan(θ) 的泰勒展開起步，這時等同針孔相機。不要用 FULL_OPENCV(會發散)。高階係數很大、正負交錯是正常的，判斷擬合好不好要看重投影誤差。
- 流程不優化主點，報告裡的 cx、cy 是先驗的值。
- 先驗模型的單位是公尺:`select-pairs`、對齊和報告的標記都用以公尺為單位的門檻。
- 觀測數少於 `--min-obs` 的影像沿用先驗姿態(`restore-prior-poses`)。只依觀測數判斷，不依位移大小：觀測數多、修正量大的影像，通常是先驗姿態的漂移被正確修正。
- 輸出在先驗模型的公尺座標系。不要寫入 SCAN_DIR;原始影像、深度和 App 輸出都要保留。輸出慣例放在 `<SCAN_DIR>_opt`。
- `prior_pose_refine.py` 只依賴 numpy 和 scipy(`requirements.txt`)。分析腳本和測試可以用 pycolmap,流程本身不行。
- Splatify-Anything 的輸出(App 識別碼 `fable-gs-capture`):`sparse/0` 等於 `poses_refined.jsonl`(App 自己優化後的姿態),比較時以它作為 App 的基準。`poses.jsonl` 是原始 ARKit,採 GL 的 camera-to-world 慣例；轉成 COLMAP 時，相機軸和世界軸都要乘上 diag(1, −1, −1)。它的內參只有針孔參數，而且對焦不一定有鎖定(部分掃描內 fx 漂移 1–2%)。

## 環境

本機路徑都放在 `local.env`,這個檔案不納入版控；換到新機器時，複製 `local.env.example` 再填入路徑。開始工作前先執行 `source local.env`:

| 變數 | 指向 |
|---|---|
| `PATH` | 加入有 CUDA 版 COLMAP 4.x 的目錄;`prior-pose-refine` 會呼叫 `colmap` |
| `PYTHON` | 有 numpy 和 scipy 的 Python(分析和測試另需 pycolmap);`prior-pose-refine` 會讀它。執行腳本時加 `-I` |
| `LICHTFELD` | LichtFeld Studio 執行檔 |
| `SCAN_ROOT` | 擷取 App 輸出的掃描 |
| `EXP_ROOT` | 實驗和修改前的備份，放在 repo 之外 |

- LichtFeld Studio 的 checkout 有未 commit 的本地 `--pose-opt` 修改：除非使用者要求，不要 stash、pull、merge 或重新編譯它，也不要更新 COLMAP。
- 執行前先確認剩餘磁碟空間：每次執行約 1.2–2 GB,確認結果後可以刪除 `work/`。
- 機器只有一張 GPU 時，計時的時候不要同時跑兩個 GPU 工作。

## 語言和回報

- 和使用者用繁體中文溝通。程式碼、註解、CLI 說明和報告輸出用英文。
- 每份文件都有英文(`NAME.md`)和繁體中文(`NAME.zh-TW.md`)兩個版本:README、AGENTS、docs/method 和 docs/benchmark。同一次修改要同時更新兩個版本，在開頭互相連結，文件內的連結要指向同一種語言。
- 簡報和報告：品質數字旁邊要列出時間成本(前處理、訓練、相對基準多花的時間);比較圖要框出明顯的差異，並說明哪一個比較好；主線要短、先講結論，細節放在有編號的附錄。避免使用拍到人或名牌的影像。
