# Baseball_Defense_Web 專案地圖

> 本檔記錄每支檔案實際在做什麼，最後更新 2026-07-06。
> 進入本專案工作前先讀此檔；改動結構或新增檔案時順手更新此檔。

## 專案目標
外野手「接殺機率模型 → OAA → 站位優化」的 Web 版重寫。
以貝氏階層羅吉斯回歸估計每顆外野飛球的接殺機率，加總成各球員的 OAA（Outs Above Average），
並用來做站位優化。這是把碩論的 `Baseball_Defense_Model_3`（已凍結為論文封存，**不可改其程式/資料**）
用全新程式碼重新實作（公式相同、程式重寫），改為直接查 PostgreSQL、不依賴預先算好的 CSV。

⚠️ **2026-07-04 起有例外**：為了能部署到免費雲端方案（`statcast` 表 5.1GB 超過免費 DB 容量上限），
`prepare_batter_balls`/`get_batter_stand`/`load_qualifying_batters`（`src/optimization.py`）
改成查 `precomputed_batter_balls`/`precomputed_batter_stand` 這兩張精簡衍生表，而不是即時查
`statcast`。這兩張表本身仍在 PostgreSQL 裡（不是 CSV），只是內容由
`scripts/precompute_batter_balls.py` 預先從 `statcast` 算好。原本直查 `statcast` 的版本在 git 歷史
（2026-07-04 之前）。改動後跟改動前的 `/api/optimize` 輸出比對過，數值差異在 1e-8 量級（浮點數
經 PostgreSQL COPY 文字格式往返造成，非行為改變）。

## 環境與慣例
- Python：`C:\Users\howard970005\AppData\Local\Programs\Python\Python313\python.exe`（直接安裝，非 venv）
- PostgreSQL：`host=localhost dbname=baseball user=postgres password=postgres`（統一定義在 `src/config.py` 的 `DSN`，2026-07-04 起所有 src/、scripts/、api/ 都從那裡 import，不再各自硬編碼）
- 溝通語言：繁體中文
- PyMC 採樣前必設 `PYTENSOR_FLAGS=optimizer_excluding=constant_folding`
  （本機 msys64 g++ 對 -inf 常數節點的 constant_folding 重寫會靜默失敗；完全停用 C 後端則慢到不可行）
- 訓練年份 `2021–2024`，`2025` 一律留作樣本外驗證（與 Model_3 的 pass18_2124 對齊，方便直接比較）

## 資料夾結構
```
data/raw/statcast/{year}.parquet           逐球 Statcast 原始資料（未篩選；2020–2025）
data/raw/positioning/{year}.parquet        球員平均起始守備站位（距離+角度；2017–2025，含內外野七個位置）
data/raw/sprint_speed/{year}.parquet       跑者速度 leaderboard（sprint_speed + hp_to_1b；2020–2025）
data/raw/savant_fielding/{year}.parquet    Savant 官方 OAA 所用的逐球守備機會（目前只有 2025）
data/reference/MLBStadiaPathData.rda       各球場外野圍牆多邊形（GeomMLBStadiums R 套件）
data/reference/batter_names.json           batter_id → 姓名快取（pybaseball 反查）
data/precomputed/re24_table.json           RE24 預期得分表（24 種壘上/出局狀況）
data/precomputed/delta_re.json             ΔRE(k, 狀態) 安打型別 × 24 狀況 × 3 型別 = 72 筆
data/precomputed/hit_type_kde.joblib       KDE 模型 P(1B|j)/P(2B|j)/P(3B|j)
figures/validation_scatter.png             Model OAA vs 官方 OAA 驗證散點圖（自用，勿放入 web）
figures/validation_scatter_v2.png          同上，改良版視覺設計（y=x 參考線＋邊際分布；自用，勿放入 web）
figures/reliability_diagram.png            Reliability Diagram（自用，勿放入 web）
models/2025/{LF,CF,RF}/                   定案模型（speed, cos, sin, fielder_dist；訓練年 2021–2024）
scripts/sql/                               四張資料表的 CREATE TABLE DDL
tests/                                      pytest 測試（51 個，`python -m pytest tests/ -v`）
                                            預設跳過標 @pytest.mark.integration 的測試（見 pytest.ini）
pytest.ini                                  定義 integration marker，預設 -m "not integration"
```
每個 model 資料夾含：`{pos}_trace.nc`（後驗樣本）、`{pos}_scaler.joblib`（StandardScaler）、
`{pos}_summary_group.csv`（群體層 mu_*）、`{pos}_summary_players.csv`（球員層係數）。

PostgreSQL 資料表：
- `statcast` — 426 萬筆逐球原始資料（2020–2025），5.1GB。只有本機訓練/評估/precompute 用，**不部署到雲端**
- `fielder_positioning` — 球員每年各位置平均站位（2017–2025；2026-07-06 起含 1B/2B/3B/SS/LF/CF/RF 七個位置，供內野擴展用）
- `sprint_speed` — 跑者速度 leaderboard（2020–2025，每年 500–630 人）。`hp_to_1b`（本壘到一壘秒數）是滾地球出局率模型的跑者端輸入，約 85% 球員有值（球級覆蓋 ~97%）
- `if_oaa_leaderboard` — 每位內野手的官方 OAA 匯總（2023–2025，每年 ~350 人）。含分位置機會數/OAA（`n_pos3~6`/`oaa_role3~6`）、方向與左右打分解。與外野的 `oaa_leaderboard` 分表，因為欄位結構不同（內野無 0–5 星概念）
- `savant_fielding` — 官方 OAA 逐球守備機會（2025，36,848 筆）
  - ⚠️ `catch_prob` 欄位是 xban（難度分數 0~1），**只有接到的球才有值**（未接球=0.000），不是所有球的接殺機率
  - ⚠️ `star_savant` 欄位是 0/1（有沒有接到），**不是** 0~5 星難度分級
- `oaa_leaderboard` — 每位外野手的官方 OAA 匯總（2025，326 人；含 0–5 星分類）
- `model_oaa` — 我方模型算的 OAA（2025，602 筆 = LF 214 + CF 172 + RF 216）
  - 欄位：`name_fielder, position, year, model_oaa, n_opp`
  - 由 `scripts/precompute_model_oaa.py` 產生，算法：對 `is_official` 子集逐球 `oaa_play=caught−catch_prob` 後加總
  - ⚠️ 系統性偏正（avg ≈ CF +4, LF +2, RF +2），根本原因是用賽季平均站位而非每球實際站位。API 端點已做中心化校正
- `precomputed_batter_balls`（2026-07-04 新增，29MB，299,345 筆）— 服務 `/api/optimize` 即時查詢，
  取代直查 `statcast`。欄位是已算好的物理衍生值（`ball_x/ball_y/flight_time/spray_angle` 等），由
  `scripts/precompute_batter_balls.py` 產生，**這張表需要部署到雲端**
- `precomputed_batter_stand`（2026-07-04 新增，560KB，8,016 筆）— 服務 `get_batter_stand`，同上由
  `scripts/precompute_batter_balls.py` 產生，**這張表需要部署到雲端**

## 資料管線（依序）
1. `scripts/fetch_statcast.py` — pybaseball 抓整年逐球 Statcast → parquet
2. `scripts/fetch_positioning.py` — 抓 Savant `/visuals/position_data` 球員平均站位（非官方 API，DevTools 逆向）。
   位置代碼用 MLB 標準編號（3=1B … 9=RF），Savant 5xx 會自動重試。
   單獨重載站位不動 statcast：`python scripts/append_years_to_db.py --positioning-only 2017 ... 2025`
3. `scripts/fetch_savant_fielding.py` — 抓 Savant 官方守備機會逐球（leaderboard + gamelogs 兩個非官方端點），
   作為「官方 OAA 所用球集」，評估時用來把樣本限制在同一母體
4. `scripts/fetch_oaa_leaderboard.py` — 抓 Savant OAA leaderboard 匯總資料 → `oaa_leaderboard` 資料表。
   每位外野手一列，含官方 OAA、全部守備機會（n_opp，含 0 星）、1~5 星的接殺數與機會數、達標旗標。
   執行：`python scripts/fetch_oaa_leaderboard.py 2025 [2024 ...]`（支援多年）。
   ⚠️ savant_fielding 的 `catch_prob`（xban）是難度分數（0=容易、1=最難），**不是**接殺機率，無法直接還原官方 OAA。
5. `scripts/fetch_if_oaa_leaderboard.py` — 抓內野官方 OAA（`/leaderboard/outs_above_average?pos=if`）→ `if_oaa_leaderboard`。
   CSV 匯出缺機會數，所以改解析頁面 `var data` JSON；頁面 year 欄為空，由腳本以請求年份補。
   執行：`python scripts/fetch_if_oaa_leaderboard.py 2023 2024 2025`
6. `scripts/fetch_sprint_speed.py` — 抓 Savant sprint speed leaderboard（含 hp_to_1b）→ parquet
   （內野滾地球模型見下方「內野擴展」章節）
7. `scripts/load_to_postgres.py` — 把 statcast/positioning/sprint_speed/savant_fielding 四種 parquet 用 COPY 灌進 PostgreSQL（可重跑，每次先 TRUNCATE 重建）

## API（FastAPI）
- `api/main.py` — FastAPI 主程式，startup 快取打者清單、姓名、各位置外野手清單、模型參數
- `api/schemas.py` — Pydantic 請求/回應模型
  - `BallPoint` 含 `responsible: str | None`（'LF'/'CF'/'RF' 或 None；catch_prob < 5% 時為 None）
  - `responsible` 目前由後端用距離算法計算（最近守備員）；見下方陷阱說明
- `api/plot.py` — 後端 matplotlib+seaborn 繪圖（重現論文 park_single v2 圖）
- 啟動：`python -m uvicorn api.main:app --port 8000`（在專案根目錄）
- **端點**：
  - `GET  /api/years`   — 可用年份清單（前端年份 tabs 用）
  - `GET  /api/batters?year=2025` — 指定年份打者（n_balls ≥ 30），回傳 `[{batter_id, name, n_balls}]`。姓名由 `data/reference/batter_names.json`（pybaseball 反查快取）提供，因 statcast.player_name 是投手
  - `GET  /api/teams`   — 30 支球隊縮寫
  - `GET  /api/fielders?min_opp=100&year=2025` — 各位置外野手清單，只含模型有 player-level 參數且該年有 `model_oaa` 紀錄的球員，依中心化後 OAA/100 降序，回傳 `{LF:[{name,oaa,n_opp,player_id,team_id}],...}`。中心化：`centered = raw_oaa − avg_oaa_per_ball × n_opp`（cross-position，去除系統偏正）；`min_opp` 可調，預設 100
  - `GET  /api/star_stats?year=2025` — 從 `oaa_leaderboard` **不限位置**按姓名查（SUM 合併多位置），回傳 `{player_name: {stars:[{opp,outs}×6], all:{opp,outs}}}`。不限位置是因為 oaa_leaderboard 記錄的位置可能跟 model_oaa 不同（例如 Langford 官方 LF、模型 CF）
  - `GET  /api/player_trend?name=...` — 球員多年 OAA 趨勢，對 `_AVAILABLE_YEARS` 逐年套同樣的中心化邏輯，回傳 `[{year, position, oaa, n_opp, rate}]`（rate = OAA/100）
  - `GET  /api/park_boundary/{team}` — 球場圍牆多邊形座標
  - `POST /api/optimize` — JSON（約 50s，no_park+with_park 各 100 起點）
    - Request: `{batter_id, on_1b, on_2b, on_3b, outs, home_team?, fielders?}`
    - `fielders`: `{LF?,CF?,RF?}` 球員名（player-level 能力）；任一指定 → positions 只回 `{"custom":...}`、圖只顯示該組**紫色星號**站位。未指定位置用聯盟平均 group mu
    - Response: `OptimizeResponse{title, situation, positions{key→PositionSet}, balls, park_boundary?, stats}`；positions key 為 `league_avg`/`no_park`/`with_park` 或 `custom`
  - `POST /api/optimize_plot` — 同 Request，回傳 JSON `{image_b64, title, situation, positions, stats}`；
    前端從 base64 解碼為 Blob URL 顯示圖，stats 數字（catch%, RE24, Δ RE24）另外用 HTML StatsPanel 呈現（不嵌在 PNG 內）

## 站位優化管線（需先 precompute，再 optimize）
1. `scripts/precompute_model_oaa.py` — 對 `is_official` 子集逐球算 `oaa_play=caught−catch_prob`（用群體層 mu_*），按球員加總後寫入 `data/precomputed/model_oaa_2025.csv`，再 UPSERT 進 `model_oaa` 表。執行：`python scripts/precompute_model_oaa.py`
2. `scripts/precompute.py` — 一次性預計算，輸出三個檔到 `data/precomputed/`：
   - `re24_table.json`：從 statcast 2021–2024 算出 24 種壘上/出局的預期得分
   - `delta_re.json`：確定性跑壘推進算出 ΔRE(k, 狀態)，3 型別 × 24 狀況 = 72 筆
   - `hit_type_kde.joblib`：KDE 模型 P(1B|j)/P(2B|j)/P(3B|j)（按打者左右手分開）
2. `src/optimization.optimize_positions(batter_id, on_1b, on_2b, on_3b, outs, ...)` — 主函式
   - 載入預計算 + 模型 → 準備打者歷史球 → 計算 w_j → L-BFGS-B multistart（極座標扇形邊界）

目標函數：θ*(s) = argmin_θ  Σ_j (1 − p̂_j(θ)) × w_j(s)
p̂_j = 1 − ∏_i(1−p_ij)（三外野手聯合接殺機率）
w_j = Σ_k P(k|j) × ΔRE(k, s)

## 核心程式（src/）
- `src/config.py` — 共用設定，目前只有 `DSN`（PostgreSQL 連線字串），所有其他模組統一從這裡 import
- `src/re24.py` — RE24 表與 ΔRE：`build_re24_table(years)` / `build_delta_re(re24, years)` / `save_re24` / `load_re24`
  - ΔRE 採**實測跑壘推進**：以安打「下一打席壘況」推算實際壘況，n<30 退回確定性 `build_delta_re_deterministic(re24)`。與論文 (Model_3) `delta_re_emp` 數值一致（72 格 |diff|<0.05）
- `src/hit_prob.py` — KDE 安打類型機率：`fit_hit_type_kde(years)` / `predict_hit_probs` / `predict_hit_probs_batch` / save/load
- `src/optimization.py` — 站位最佳化：`optimize_positions(batter_id, on_1b, on_2b, on_3b, outs, ..., home_team=None)`
  - `home_team`（可選）：指定球場縮寫（如 'BOS'）→ 過濾在該球場會打牆的球，排除出最佳化
  - 搜尋範圍：**極座標扇形**（`_POLAR_BOUNDS`）：LF r∈[150,400]ft / θ∈[-45,0]°、CF r∈[150,400]ft / θ∈[-22.5,22.5]°、RF r∈[150,400]ft / θ∈[0,45]°（所有球場相同）
  - θ 從 y 軸量起（指向中外野=0），正值朝右，負值朝左；確保守備員留在 fair territory（foul line ≈ ±45°）
  - 最佳化器：**L-BFGS-B multistart**（預設 `n_restarts=20`，2026-07-05 從 100 降低，見下方說明），在正規化後的 [0,1]^6 空間運作。目標函數非凸、多 local optima（emp ΔRE 後 2B 權重大、地形崎嶇），起點不足會卡 local（歷史上曾記錄 20 起點對大谷卡在 RF 超淺解的案例，但那是舊資料/設定下的觀察，跟下方 2026-07-05 的重新測試條件不同，不能直接套用）；歷史上實測 100/300 起點穩定收斂全域最優
  - 收斂率歷史實測：26/30 找到全域最優解（舊資料/設定）
  - ⚠️ **2026-07-05：`n_restarts` 預設值從 100 降到 20**（為了縮短 Render 免費方案 0.1 CPU 上 `/api/optimize` 的耗時，原本 100 restarts 要 4分20秒）。
    實測單次 `optimize_positions`（無併發競爭）約 44~50 秒，不是原本估計的「約 40 秒」；細節與併發影響見下方「部署上線」章節「已知效能限制」。
    - **第一輪驗證（樣本太小，結論有誤導性）**：用單年資料 + `seed=42` 測過 17 個真實打者（15 個隨機抽樣＋大谷翔平＋Kwan, Steven），`n_restarts=20` vs `100` 的 objective 0 個不一致，一度誤以為 20 完全安全。
    - **第二輪驗證（換一批 25 個新的隨機打者，`random.seed(1)`）打臉了這個結論**：
      | n_restarts | 平均耗時 | 跟 `n_restarts=100`（視為真解）不一致比例 |
      |---|---|---|
      | 20 | 0.57s | **3/25（12%）** |
      | 10 | 0.37s | 5/25（20%） |
      | 5  | 0.27s | 12/25（48%） |
    - **結論（2026-07-05 更新）**：`n_restarts=20`（目前線上預設值）本身就有約 12% 機率收斂到非全域最優解，不是之前以為的 0%。繼續往下調（10、5）失敗率會更高，**不建議再往下降**。之前「0/15」的驗證只是抽樣運氣好，不能代表真實安全性——這是一個提醒：小樣本（<20）的收斂穩定性測試容易產生錯誤的安全感，之後如果要再調這個參數，至少要用 25+ 個不同 seed 抽樣的打者才有代表性。
    - 目前線上維持 `n_restarts=20`（速度與正確性的權衡下暫時可接受），若要同時兼顧速度與正確性，下一步應該是升級 Render 付費方案（Starter $7/月，5倍 CPU）換取能調回更高 `n_restarts` 的空間，而不是繼續往下犧牲正確性。
  - **2026-07-05 新增 warm start**（`warm_start_xy` 參數）：`with_park` 呼叫時用 `no_park` 的解頂替其中一個隨機起點（不是額外多跑一次，總 evaluate 次數仍是 `n_restarts`，零額外算力成本）。
    - **第一次驗證（20 樣本）**：`n_restarts=20` 時 miss rate 跟純隨機 baseline 完全一樣（2/20），沒有退步，但也沒有加速（`n_restarts` 沒變）。
    - **第二次驗證（30 個新的隨機打者/壘況，`random.seed(123)` 選樣，`home_team='BOS'`，對照 `n_restarts=150` 真解）**——這次改成同時掃描不同 `n_restarts` 值，找出warm start 到底能不能安全降低 restart 數：

      | n_restarts | miss（無 warm start） | miss（+warm start） | 平均耗時（本機） |
      |---|---|---|---|
      | 20（原本線上值） | 3/30 (10%) | 3/30 (10%) | 0.406s |
      | 15 | 5/30 (17%) | 3/30 (10%) | 0.306s |
      | 12 | 6/30 (20%) | 3/30 (10%) | 0.243s |
      | 10 | 6/30 (20%) | 3/30 (10%) | 0.203s |
      | **8** | 7/30 (23%) | **3/30 (10%)** | **0.159s** |
      | 6 | 8/30 (27%) | 4/30 (13%) — 開始惡化 | 0.126s |
      | 5 | 8/30 (27%) | 5/30 (17%) | 0.099s |

      **結論：`with_park` 的 `n_restarts` 可以從 20 降到 8，warm start 加持下 miss rate 完全不變（3/30，跟 20 一樣）**，降到 6 才開始惡化。3/30 的 miss 是幾個天生難收斂的打者/壘況組合，restart 數在 8~20 之間對它們沒差；warm start 補的是「restart 數變少時才會冒出來的額外不穩定」。
    - **已套用**：`api/main.py` 的 `with_park` 呼叫改成 `n_restarts=8`（連同 `warm_start_xy`）。`no_park`（沒有warm start可用）與 `custom`（指定外野手）維持原本 `n_restarts=20` 不變，只有 `with_park` 這個特定呼叫調低。本機測約快 2.5 倍，換算到 Render 免費方案的 44~50s 應可縮到約 18~20s 左右（實際數字仍要等 timing log 驗證）。
  - **2026-07-05 隨機起點改用 Latin Hypercube Sampling（LHS），但只在沒有 warm start 時用**：
    起初想全面把均勻隨機起點換成 `scipy.stats.qmc.LatinHypercube`（同樣起點數下，每一維度邊際分布涵蓋更均勻），
    30 樣本驗證發現**兩種情境效果相反**：
    - **`no_park`（無 warm start）**：LHS 明顯比均勻隨機好。`n_restarts=20` 時 miss rate 從 4/30（均勻隨機）
      降到 2/30（LHS），`n_restarts=10` 時 LHS 是 3/30——比均勻隨機的 20-restart baseline（4/30）還好，
      等於**同時**拿到更高正確性跟 2 倍加速。
    - **`with_park`（用 `no_park` 解 warm start，`n_restarts=8`）**：直接在同一批 30 個樣本上做受控比較
      （只換 sampler，其他完全一樣）——**LHS 反而更差**：均勻隨機 4/30、LHS 6/30。推測原因：LHS 的分層
      設計是針對「這批起點」整體去算的，硬插入一個外部給的 warm start 點會打亂分層假設的均勻覆蓋；
      均勻隨機沒有這種對「起點組合完整性」的依賴，跟外來的 warm start 點混在一起不會有結構性衝突。
    - **結論並已套用**：`src/optimization.py` 的 `optimize_positions` 內部依 `warm_start_xy` 是否提供
      自動切換——**沒有 warm start 用 LHS，有 warm start 用均勻隨機**，呼叫端不需要關心這個細節。
      同時把 `api/main.py` 的 `no_park` 呼叫加上 `n_restarts=10`（原本 20，因為現在自動用 LHS，
      不需要再測 `_sampler` 參數，該參數已在正式程式碼移除）。`custom` 模式維持 `n_restarts=20`
      不變（沒有測試過調低，但因為也沒有 warm start，會自動受益於 LHS 的正確性提升，成本不變）。
  - **2026-07-05 放寬 L-BFGS-B 收斂容忍度**（原本 `ftol=1e-10, gtol=1e-6`）：先做診斷確認 `maxiter=500`
    從未被打到（5 個打者×20 restarts 抽樣，迭代次數中位數只有 15、最大 27），代表真正限制運算量的是
    ftol/gtol 這兩個收斂門檻，不是 maxiter。用 30 樣本在**正式的 production 設定**上驗證（`no_park`
    `n_restarts=10`+LHS、`with_park` `n_restarts=8`+warm start，對照各自的 `n_restarts=150` 真解）：

      | 容忍度設定 | no_park miss | no_park 耗時 | with_park miss | with_park 耗時 |
      |---|---|---|---|---|
      | 原本（ftol=1e-10, gtol=1e-6） | 4/30 | 0.180s | 7/30 | 0.147s |
      | **1e-6 / 1e-4** | **4/30（一樣）** | **0.154s（快14%）** | **7/30（一樣）** | **0.122s（快17%）** |
      | 1e-4 / 1e-3 | 7/30 — 變差 | 0.129s | 9/30 — 變差 | 0.102s |
      | 1e-3 / 1e-2 | 20/30 — 大幅變差 | 0.101s | 17/30 — 大幅變差 | 0.082s |

      ⚠️ 一開始只用單一 batter 的單次 restart 做診斷（`n_restarts=20`，寬裕的重複次數下）看起來放寬到
      1e-4/1e-3、甚至 1e-3/1e-2 都不影響最終解——但那是因為 20 個 restart 夠多，個別 restart 精度
      降低也有其他 restart 能補上。在**正式較低的 `n_restarts`（10、8）**下重新驗證，才發現放寬過頭
      miss rate 會跳升到 60%+——`n_restarts` 降低和收斂容忍度放寬兩個「用正確性換速度」的手段疊加時，
      風險不是線性相加，一定要在實際要用的 `n_restarts` 下驗證，不能只在寬裕的 restart 數下測試就外推。
    - **已套用**：`ftol=1e-6, gtol=1e-4`（`maxiter` 維持 500 不變，反正從未被打到）。跟原本 tight
      設定 miss rate 完全一樣，`no_park`/`with_park` 各快 14~17%。這個改動影響所有呼叫（`no_park`/
      `with_park`/`custom`），不只是低 restart 數的路徑。
  - 回傳含 `n_wall_balls`（被排除的打牆球數）
- `src/stadium_walls.py` — 球場圍牆多邊形：`is_wall_ball(x, y, home_team)` 判斷落點是否在球場外
  - 資料來源：`data/reference/MLBStadiaPathData.rda`（GeomMLBStadiums R 套件，178 KB）
  - `SUPPORTED_TEAMS`：30 支 MLB 球隊縮寫清單
- `src/physics.py` — 純物理/幾何工具：像素座標→本壘原點直角座標、拋體飛行時間、
  站位極座標→直角座標、相對「背向本壘」跑動角（0=正前 charge，±π=正後 retreat）、6 區方向 zone、接殺事件判定。
  公式重寫自 Model_3 的 `src/core/physics.py`。
- `src/defender_features.py` — 主特徵工程。查 statcast JOIN fielder_positioning，
  篩選（type='X'、對應 hit_location、bb_type∈{fly_ball,line_drive}、排除 Strategic/4th outfielder、非全壘打），
  算出 `flight_time / required_speed / fielder_dist / cos_angle / sin_angle / zone / caught`。
  - `get_defender_opportunities(position, year)` → 單位置單年的有效守備機會 DataFrame
  - `mark_official(df)` → 比對 savant_fielding（key=fielder_id+game_pk+at_bat_number）標 `is_official`，
    評估時只取 `is_official` 子集，與官方 OAA 同母體
  - ⚠️ statcast 的 `player_name` 欄位是**投手**不是守備員；球員層分組一律用 `name_fielder`
- `src/model_training.py` — **定案模型**。特徵 `speed, cos_angle, sin_angle, fielder_dist`（全部有球員層隨機效應）。
  非中心化參數化，NUTS（draws/tune=2000, chains=4, target_accept=0.95, seed=42）。
  - speed = 接殺難度（需要跑多快），dist = 時間充裕度（同 speed 下距離越遠代表飛行時間越長，越從容）
  - 特徵設計選 dist 而非 time：避免 speed+time 的三角共線（speed=dist/time），VIF 全 < 1.5

## 訓練與評估腳本（scripts/）
- `train_dist.py [LF CF RF]` → `models/2025/`（定案模型；內部 import `src.model_training`）
- `evaluate_2025.py` — 2025 樣本外 OAA 相關係數評估（official 子集）：**R=0.7952**
  做的事：取 2025 `is_official` 子集 → 用**群體層後驗平均 mu_*** 算 catch_prob
  → 逐球 OAA=caught−catch_prob → 加總到球員 → 與 `oaa_leaderboard`（is_qualified=True，n=89）算 Pearson R
- `make_validation_plot.py` — 繪製 Model OAA vs 官方 OAA 驗證散點圖 → `figures/validation_scatter.png`（自用）
- `make_validation_plot_v2.py` — 同上資料/計算邏輯，改良視覺設計版：等比例軸＋y=x 完美一致參考線（揭露 model OAA
  數值尺度比官方大，slope≈0.41、model SD≈14 vs 官方 SD≈7，只讀原本兩支腳本沒呈現的資訊）、上/右邊際分布直方圖、
  Okabe-Ito 色盲友善配色 → `figures/validation_scatter_v2.png`（自用，不覆蓋/不修改 v1）
- `make_reliability_plot.py` — 繪製 Reliability Diagram → `figures/reliability_diagram.png`（自用）
  使用 2025 全部球（**不**限 is_official）：n=52,598，Brier=0.0632，LogLoss=0.2142，MAE=0.1278

## 內野擴展（2026-07-06 起，開發中）

把站位最佳化延伸到內野滾地球。與外野的根本差異：出局是「攔截＋傳球 vs 跑者到一壘」的
競速（外野只需接到球）；滾地球 hc_x/hc_y 是被處理位置非落點（出局球記錄深度中位數 ~46
savant units vs 安打 ~91），所以位置資訊只能用 spray angle（1D），不能像外野建 2D 落點分布。
訓練資料從 2023 起（禁趨位後規則同代），2025 留樣本外驗證。

- `src/if_dataset.py` — 滾地球特徵工程（設計依據：Melville 2024 §3.1 的 a_d/b_t、
  Tango 2020 Infield OAA 的競速結構）。`build_gb_dataset(years, bases_empty, alignment)`
  一站式產出；特徵計算在純函式 `attach_features()`（可測試，不碰 DB）
- `src/if_model.py` — **兩個模型、兩種角色**（這是關鍵設計決策，勿混用）：
  - 優化用 GLM（`FielderGeometryFeatures` + logistic）：只用野手相對幾何，**刻意排除
    raw spray 項**——spray 的位置特定出局率模式反映「現在聯盟都站哪」（內生性），搬動
    野手後不會保留，混入會讓優化器重複計算
  - 評價用 GBM：spray+球質+跑者、無任何野手資訊的聯盟平均難度模型（xBA 式 p̂）
  - 依據（2026-07-06 消融，train 2023→val 2024）：GBM「球質+跑者+raw spray」AUC 0.8065
    ≈ 全特徵 GBM 0.8055，野手特徵在其上**零增益**——Standard 佈陣下賽季平均站位近似
    spray 的確定函數，位置訊號吃掉了一切；野手相對 GLM 的價值是 counterfactual 結構
    不是 AUC
  - 交互作用配置用 train 2023→val 2024 選定（tensor ad×bt、ad×EV、hp×throw、hp×bt），
    2025 只在最終評估碰一次
- `scripts/train_if_gb.py` — 訓練（2021–2024）＋樣本外評估（2025）→ `models/if_gb/`
  - 主範圍「無人在壘 + Standard 佈陣」（Melville 同樣排除壘上有人；1B hold runner 會拉動站位）
  - **2026-07-09 現行結果（訓練改 2021–2024，n_train=73,379、n_test=18,404）**：優化用 GLM
    AUC=0.7531、Brier=0.162、校準最大偏差 0.031；評價用 GBM AUC=0.8165、Brier=0.139、
    校準最大偏差 0.020
  - （歷史對照 2026-07-06，訓練 2023–2024：GLM AUC=0.754/校準 0.029；GBM AUC=0.815/
    校準 0.026——擴充訓練年份差異在雜訊內，見下方「訓練年份實驗」）
- `scripts/evaluate_if_2025.py` — 階段 2 球員評價：difficulty GBM 當 p̂（2021–24 訓練、
  2025 評分，無球員資訊→無循環論證），球員 model OAA = Σ(is_out − p̂)，對照
  `if_oaa_leaderboard` 官方數字
  - 歸責規則：出局球給實際處理者（`hit_location` 3–6，92.5% 的出局球適用，其中 26.9%
    跟最近角距不同——改用 hit_location 讓 qualified R 從 0.48 → 0.53）；安打球與投手/
    捕手處理的球退回最近角距內野手
  - 分位置中心化（官方是「跟同位置平均比」）：每球 oaa_play 減去歸責位置的平均
  - **2026-07-09 現行結果（訓練 2021–2024，2025 樣本外，qualified n=158）**：Pearson R=0.521、
    Spearman=0.555、每球率 R=0.585；分位置（n≥100）1B 0.68 / 2B 0.65 / 3B 0.61 / **SS 0.33**
    （SS 對實際起始位置最敏感，賽季平均站位在此損失最大）；scale 健康（model SD 8.3 vs
    官方 6.9，不像外野 2–3 倍——因為 p̂ 是聯盟平均難度不是站位相依接殺率）
  - （歷史對照 2026-07-06，訓練 2023–2024：R=0.525/Spearman 0.562/每球率 0.591，
    差異在雜訊內）
- `src/if_optimize.py` — 禁趨位約束下的四內野手站位優化（階段 3+4）
  - 打者分布直接用歷史滾地球（角度不受站位污染），不需要 KDE
  - 規則約束全部化成 box bounds：1B/2B 角度 [1°,44°]、3B/SS [-44°,-1°]（兩側各兩人、
    不可換邊）；深度重參數化為「MIN_DEPTH 到內野土外緣的比例」，土外緣 =
    以投手板為圓心半徑 95 呎的弧 r_max(θ)=60.5cosθ+√(95²−(60.5sinθ)²)
  - MIN_DEPTH=75 呎：低於訓練支撐範圍屬 GLM 外插，實測 60 呎時優化器會把打者冷區的
    「閒置野手」停在外插假象位置
  - 同側標籤可互換 → 回傳前正規化（角落位置靠邊線）
  - LHS 多起點 + L-BFGS-B（重用外野的優化模式）；`geometry_features()` 與
    `if_dataset.attach_features()` 公式一致性有測試把關
- `scripts/optimize_if_demo.py` — demo：2023–24 滾地球最多的打者，最佳化 vs 聯盟平均站位
  - **2026-07-06 結果**：期望出局率增益 +0.011~+0.029（每 450 顆滾地球約 +5~13 個出局），
    解符合直覺（右打：3B 顧線+SS 顧洞+右側收中間的合法 shade）
- `scripts/validate_if_positioning.py` — 階段 5 跨年驗證 → `models/if_gb/validation_2025.json`
  - 對每位合格打者（train GB≥150、test GB≥80，n=212）：跨年增益 =「2023–24 球優化的站位」
    vs 聯盟平均，**兩者都評估在 2025 球上**；同年上限用 2025 球自己優化再評 2025（同樣
    樣本，可直接比較）。成效是 optimizer GLM 模型評分——反事實站位下的實際出局不可觀測，
    此設計檢驗的是「打者傾向與模型的跨年可遷移性」
  - **2026-07-07 結果**：跨年增益平均 +0.0155/GB（每 450 GB ≈ +7.0 出局）、同年上限
    +0.0243、保留率（Σ跨年/Σ同年）**64.0%**、96.2% 打者正增益；t=21.50、p=4.4e-55；
    左右打接近（L +0.0162 / R +0.0152）
  - 逐打者結果落盤 `validation_rows_2025.csv` 當 checkpoint，中斷後重跑同指令即續算
- 官方對照資料：`if_oaa_leaderboard` 表（見資料表清單）
- 已知限制：站位仍是賽季平均（同外野的 OAA scale 問題，內野對站位誤差更敏感——反應時間
  僅 1–2 秒）；跑者速度用賽季平均 hp_to_1b（官方 OAA 也是用平均 sprint speed，做法一致）

### 內野 web 整合（2026-07-09）

**核心設計決策：內野優化全部離線預算，線上只查表**。外野 `/api/optimize` 能即時算是因為
目標函數是向量化 numpy；內野 GLM 每次評估都過 sklearn pipeline，即使做了快速路徑
（見下）本機 `n_restarts=20` 仍要 ~4 秒，Render 0.1 CPU 上不可行。

- `src/if_optimize.py` 的 `_FastGLMObjective`：把 GLM pipeline 展開成純 numpy——
  不隨站位變動的項（launch_angle spline、EV/hp_to_1b z 分數、stand、截距）每位打者算一次，
  每次評估只重算 ad_min/ball_time/throw_dist 相關項。**加速約 10 倍**（n=20 從 43s → 4.2s）。
  逐點與 `model.predict_proba` 等價到 1e-10（有測試）；`optimize_infield` 自動偵測
  pipeline 走快速路徑，其他模型（如測試的 DummyModel）退回通用路徑。
  ⚠️ 浮點最後一位的差異會讓 L-BFGS-B 在平坦地形偶爾走到不同局部解（6 組對照中 2 組
  exp_outs 差 ~3e-5，遠低於 1e-4 容忍值），屬預期非 bug
- 收斂穩定性測試（`scripts/test_if_convergence.py`，30 位打者、參考解 n_restarts=150）：
  n=20 中位落後 0.00003、容忍 1e-3 時 miss 1/30；n=12 以下 miss rate 20%+。
  結論：**離線預算用 50+，線上即時算（如果未來要做）n=20 是下限**。
  checkpoint `models/if_gb/convergence_rows.csv`
- `scripts/precompute_if_optimize.py`：對每個 (打者, 年份)（2023–2025、該年 GB≥50，
  各年約 390 位）用 n_restarts=50 優化，寫入 `precomputed_if_positions`（站位+期望出局率）
  與 `precomputed_if_gbs`（逐球 spray/ball_x/ball_y/EV/is_out/兩組站位下的 P(out)，
  前端畫點與上色用）。逐 (打者, 年份) checkpoint（positions 列是 commit marker，
  續跑時清孤兒 balls 列）；聯盟平均站位存 `data/precomputed/if_league_positions.json`
  （API startup 讀）。`--target-dsn` 同 precompute_batter_balls 的雲端部署模式；
  `--refresh-gbs` 只重算逐球表（沿用 DB 已存站位，約 30 分鐘）——改逐球表 schema 或
  展示欄位時用，不必重跑 3 小時優化
- **hc 座標語意（2026-07-09 查證）**：2024 滾地球出局且 hit_location=3~6 的球（n=36,547），
  hc 換呎（×2.5）後深度中位 118 呎（內野手深度帶）、距歸責野手賽季平均站位中位 32 呎
  （P10-P90 15~55）；一壘安打深度中位 224 呎（外野撿球帶）、P10=70 呎（內野安打）。
  結論：hc ≈ 球被處理/撿起的位置附近（精確語意無法從資料分辨），但**確定是結果與守備的
  函數**（同一顆球出局 vs 穿出去座標差一倍）→ 只能展示，不能拿來建打者分布（角度除外）
- `scripts/precompute_if_model_oaa.py`：排名頁資料 → `if_model_oaa` 表（2025 樣本外，
  349 位）。計算邏輯與 evaluate_if_2025.py 共用 `src/if_eval.py`（GBM 評分＋hit_location
  歸責＋分位置中心化），重構後驗證所有評估數字不變（qualified R=0.525 等）
- API 端點（全部查表即回，無運算；startup 快取，缺表不中斷啟動——Neon 還沒 sync 時
  內野端點回空/404）：
  - `GET /api/if_years` — precomputed_if_positions 有的年份
  - `GET /api/if_batters?year=` — 該年打者清單（batter_id/name/n_gb/stand，n_gb 降序）
  - `GET /api/if_result?batter_id=&year=` — 聯盟平均+最佳化站位（角度/深度/xy）、
    逐球資料、期望出局率與增益（gain×450 = 一季規模的出局數）
  - `GET /api/if_fielders?year=2025&min_balls=100` — 排名（model OAA 已分位置中心化，
    不需要外野那種跨位置校正），LEFT JOIN if_oaa_leaderboard 帶官方 OAA 對照欄
- 前端（NavBar 四連結：外野站位/內野站位/外野排名/內野排名；連結多了在窄螢幕改橫向捲動）：
  - `/infield`（`pages/Infield.jsx`）— 鏡射主頁版型：年份 tabs → 打者搜尋 → 顯示按鈕（瞬間，
    無壘況/球場/守備員選項——模型範圍是無人在壘+Standard、滾地球與球場無關、GLM 無球員層參數）。
    `components/InfieldChart.jsx`：SVG 鑽石場地（土外緣弧=優化器同一公式）、聯盟平均（藍菱形）
    vs 最佳化（紫星）、滾地球畫在 **Statcast 記錄座標**（ball_x/ball_y；出局≈處理位置、
    安打≈撿球位置，legend 旁有說明文字，語意查證見上方「hc 座標語意」）、視野涵蓋到
    撿球深度（±240×320 呎，更深的球沿同方向夾回邊緣）、RdYlGn 按 P(out) 上色
    （可切平均/最佳化站位）、P(out) 範圍滑桿、hover tooltip
  - `/if-rankings`（`pages/InfieldRankings.jsx`）— 鏡射排名頁版型（tabs ALL/1B/2B/3B/SS、
    球隊篩選、min balls 滑桿、可排序），欄位=模型機會/模型OAA/OAA/100+官方三欄
    （內野無星級概念，以官方 OAA 對照取代星級分解）；只有 2025（樣本外年），無多年趨勢 modal
  - `components/playerDisplay.jsx` — 從 Rankings.jsx 抽出的共用元件（頭像/隊徽/配色）
- **部署（Neon）需要 sync 的表**：`precomputed_if_positions`、`precomputed_if_gbs`、
  `if_model_oaa`、`if_oaa_leaderboard`（排名頁 JOIN 用；2026-07-09 已全部同步過一輪，
  逐球表含 ball_x/ball_y 的新 schema 也已同日重灌並驗證 142,189 列全帶座標）。
  sync 方式見「部署上線」章節。陷阱：nullable INTEGER 欄 pandas 會讀成 float，
  COPY 前要轉 Int64
- **訓練年份實驗（2026-07-09）**：使用者提議訓練改用 2021–2024（Standard 子集篩選下
  shift 時代資料理論上可用）。實驗結果（同一 2025 樣本外）：GLM AUC 0.7531 vs 現行
  0.7540、GBM 0.8165 vs 0.8150、球員評價 R 0.521 vs 0.525——資料翻倍後三指標全在
  雜訊內，學習曲線已飽和、2021–22 的賽季平均站位污染（聯盟平均在 2022→2023 有
  斷點：3B/SS 角度外移 2~3°、2B 深度縮 4 呎）幅度不大。**2026-07-09 已正式改用
  2021–24**：train_if_gb.py / evaluate_if_2025.py / precompute_if_model_oaa.py 三處
  TRAIN_YEARS 同步改（後兩者是內部重訓 GBM，不載 joblib）→ 重訓（數字與實驗一致）→
  重跑 precompute 全下游 → Neon 再 sync。跨年驗證（validate_if_positioning.py）的
  TRAIN_YEARS 是打者分布年份（驗證設計），維持 2023–24 不隨訓練年份改。
  後續還計劃加壘況/出局數情境（階段A=RE24 加權+線上即時算
  +無人在壘解 warm start；階段B=有人在壘 out 模型 force/DP/hold runner，屬新研究）
- **內野球員層（野手個人化站位）驗證為無訊號，已收案（2026-07-10）**：目標是對齊外野的
  「指定野手 → 站位建議跟著變」。前提檢定：能力必須與幾何交互才會改變 argmax（純截距
  不動最佳解）。用 ad_eff = ad_min × scale(能力)^γ 掃描 γ，兩種能力 proxy 都是 γ=0 最佳
  且單調變差：①sprint speed（train 2021–23→val 2024，league/within 兩變體，
  `scripts/exp_if_speed_scaling.py`）②前一年官方 OAA rate 含橫向分解
  （球 2024←proxy 2023→球 2025←proxy 2024，無洩漏，`scripts/exp_if_oaa_scaling.py`）。
  解讀：野手整體轉換力確實有差（評價 R=0.52），但那是截距性質（手套/轉傳/站位品質）；
  會改變站位的「橫向 range × 幾何」成分在 MLB 選材壓縮（sprint P10–P90 僅 ±6%）＋
  賽季平均站位誤差（ad_min 本身帶噪，同 SS R=0.33 根源）下量測不到。
  ~~結論：公開資料下不值得投入~~（使用者知情後仍決定對齊外野架構，後續實作推翻了
  proxy 結論的一半，見下一條）。
- **內野貝葉斯球員層（2026-07-10，使用者決定對齊外野方式）**：`scripts/train_if_bayes.py`
  （73k 球、7hr MCMC、r_hat 1.000）。最近野手歸責、群體層=GLM 完整設計矩陣（品質無損：
  2025 AUC 0.7533 vs GLM 0.7531）、球員層=隨機截距 alpha＋ad_z 隨機斜率 g（非中心化）。
  **關鍵發現：直接從轉換資料估的球員效應有跨年訊號**（+alpha logloss −0.0008、
  +g −0.0004、合計 AUC 0.7545）——γ proxy 實驗的陰性是 proxy 太弱，不是效應不存在。
  shrinkage 比值 alpha 0.56/g 0.63（外野 beta_dist 0.65 同級）；alpha vs 官方 OAA rate
  r=0.225。產物（scripts/export_if_bayes.py）：後驗平均匯出成 sklearn pipeline
  （drop-in 換 GLM，等價 2e-16）＋ IF_player_effects.csv（578 位野手 alpha/g）。
  優化器：`optimize_infield(..., player_effects=)`，效應加在最近野手、個人化時槽位
  綁定不重排（tests 9/9）。**量級實驗**（exp_if_personalized_positions.py v2，
  25 打者）：外插診斷乾淨（ad_min 位移 <0.2°，優化器沒鑽 g·ad_z 線性外插漏洞）；
  但 repositioning gain 中位僅 +0.0002~+0.0006/GB（450GB ≈ +0.1~0.3 出局/季），
  站位移動中位 23-28 呎是**平坦地形的等值漂移**（gain 小、移動大 = 高原上換點）。
  含意：個人化站位建議的「視覺差異」大部分是解的不唯一性，不是野手能力的實質後果；
  web 呈現建議用零效應解 warm start 的錨定式個人化（位移只反映效應的真實拉力）。
  v1 教訓：**不同效應設定的 exp_outs 不可直接相減**（g 的 ad_z 中心化整體平移水準，
  會出現好陣容期望出局率反而低的假象），要在同一效應模型內比較

## 前端（React + Vite）

路由（react-router-dom）：
- `/` — 外野手站位最佳化（主頁），左側面板控制；右側顯示站位圖
- `/rankings` — 外野手守備排名頁

共用元件：
- `src/components/NavBar.jsx` — 頂部持久導覽列（sticky），品牌名「MLB Lab」，連結「外野手站位最佳化」/「守備排名」，當前頁高亮
- `src/components/Layout.jsx` — 包住所有頁面，注入 NavBar；背景 `#f1f5f9`
- `src/components/SearchSelect.jsx` — 可打字搜尋下拉
- `src/components/GameStateForm.jsx` — 壘上跑者 + 出局數
- `src/components/SprayChart.jsx` — 互動落點圖（SVG）
  - RdYlGn 顏色 by catch_prob，或切換為 LF/CF/RF 責任歸屬顏色（owner mode）
  - 接殺機率範圍滑桿（probMin/probMax），過濾顯示球數
  - 點擊守備員標記 → 高亮該守備員負責的球，其餘球淡化至 10% opacity
  - 責任歸屬（`responsible`）由後端提供；若無則前端 fallback 用距離
  - KDE 等高線（Marching Squares + feGaussianBlur 平滑）
- `src/components/DensityChart.jsx` — 落點密度畫布（canvas radial gradient，非等高線）

App.jsx（`/`）：
- 左側面板（280px，手機版見下方響應式說明）：頂部年份 tabs → 打者搜尋 → 比賽狀況 → 球場 → 外野手（min_opp 滑桿＋守備員下拉）；底部 footer 放「⇔ 比較模式」切換 + 計算按鈕
- 比較模式：A/B 各自獨立選球場（homeTeam / homeTeamB）與外野手，並排顯示兩張圖 + CompareStats 表（含座標差 Δ(dx,dy)）
- 圖片右上角有「↓ 下載」按鈕（Blob URL download）
- 空白狀態顯示引導文字（無圖示）
- 圖模式 toggle：**落點密度圖**（matplotlib PNG）/ **互動圖**（SprayChart SVG）

Rankings 頁（`src/pages/Rankings.jsx`）：
- 頂部：年份選擇 → 球隊篩選（全部球隊 / 各隊縮寫，年份切換時重設）→ min_opp 滑桿（不重設球隊篩選）
- LF/CF/RF tab 切換
- 欄位：# / 球員（圓形頭像＋球隊 logo）/ 模型機會 / 模型OAA / OAA/100 / 5 Star(0-25%) Outs-Opp-% / 4 Star / 3 Star / 2 Star / 1 Star / All Plays
- 所有欄位可點擊排序（升/降序），排序欄換淡藍底色
- 球員名字可點擊：開啟多年趨勢 modal（SVG 折線圖，X軸=年份、Y軸=OAA/100，LF藍/CF綠/RF橘；位置標籤顯示在 modal 標題旁）
- 球員頭像與球隊 logo 透過 `_team_map` 取得（startup 時從 MLB Stats API 載入，不受 min_opp 限制）
- 括號內 OAA 標「模型估計，非 Statcast 官方」

### 響應式/手機版（2026-07-06）
專案原本完全沒有 mobile breakpoint，用 Playwright 在 375×812 實測後修的問題：
- **`App.jsx` 主頁最嚴重**：左側 280px 固定寬面板 + `display:flex` 無 `flex-direction` 響應，手機上右側圖表區被擠到剩不到 100px 寬，空白狀態文字整個變成一行一個字的直排。
  修法：`.app-body`/`.app-panel`/`.app-chart-area`/`.compare-row` 這幾個 class 配合 `App.jsx` 底部 `<style>` 內的 `@media (max-width: 768px)`，窄螢幕下改成 `flex-direction: column`（面板疊在圖表區上方，寬度變 100%）。因為原本用 inline style 設定 `width`/`minHeight`/`border-right`，媒體查詢要贏過 inline style 必須加 `!important`（唯一乾淨解法，不是隨便亂加）。
- **`Rankings.jsx` 標題擠壓**：`<h1>` 跟年份 tabs 同一個 flex row、沒有 `flexWrap`，窄螢幕下標題被擠到跟上面同樣的直排問題。修法：容器加 `flexWrap:'wrap'`，`<h1>` 加 `flexShrink:0`（保證標題永遠不被壓縮，擠不下就讓 tabs 換行，不是壓文字）。
- **`Rankings.jsx` 多年趨勢 modal 在 375px 螢幕會溢出**：原本 `minWidth: 380`（比 375px 螢幕還寬），改成 `width: 'min(380px, 92vw)'`。
- **`TrendChart`（modal 裡的 SVG 折線圖）原本用固定 `width={400} height={200}`，不會隨容器縮放**（跟 `SprayChart`/主頁的 matplotlib 圖不同，那兩個本來就是響應式的）。改成 `viewBox` + `style={{width:'100%', height:'auto'}}`，比照 `SprayChart` 的做法。
- Rankings 頁的排名表格本來就有 `overflowX:'auto'` 包住、篩選列本來就有 `flexWrap`，這兩處沒有額外的手機版問題。
- 已用 Playwright 實測 375px（iPhone 標準寬）、320px（iPhone SE，最窄常見寬度）、1440px（桌機迴歸測試）三種寬度，主頁單張模式、比較模式、Rankings 列表與 modal 都過。

## 部署上線（2026-07-05）

**線上網址**：https://baseball-defense-web.onrender.com（前端 + API 同一個服務）

**架構**：
- Render 單一 Web Service（免費方案，0.1 CPU）：Build Command 先 `pip install -r requirements.txt`
  再 `cd frontend && npm install && npm run build`；Start Command `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
  `.python-version`（內容 `3.13`）鎖定 Python 版本，避免抓到太新版本導致套件從原始碼編譯失敗
- `api/main.py` 檔案最後：若 `frontend/dist` 存在就掛載成靜態檔案（見 `_FRONTEND_DIST` 那段），
  本機開發沒 build 前端時完全不影響，只有部署時才會啟用
- 資料庫：Neon（免費方案，region ap-southeast-1 Singapore）。環境變數 `BASEBALL_DSN` 覆寫 `src/config.py` 的 DSN
- Neon 上**只有**這幾張表：`model_oaa`、`oaa_leaderboard`、`fielder_positioning`、`model_star_stats`、
  `precomputed_batter_balls`、`precomputed_batter_stand`。**沒有** `statcast`（5.1GB，超過免費方案容量，
  只存在本機，訓練/評估/precompute 都只能在本機跑）、也沒有 `savant_fielding`（只有評估腳本用得到）

**更新雲端資料的方式**（例如匯入新年份的 statcast 之後）：
1. 本機重跑 `python scripts/precompute_batter_balls.py`（source=target=本機 DSN，更新本機的兩張精簡表）
2. 需要把新資料同步到 Neon 時，本機讀出 dataframe 後用 `psycopg2.connect('')`（空字串，靠 `PG*` 環境變數連線，
   見下方「Windows/Git Bash 已知問題」）連到 Neon 寫入，或用 `pg_dump`/`psql` 匯出匯入四張小表
3. 改完 `src/`/`api/` 程式碼後：`git add` → `git commit` → 用 **GitHub Desktop** push（見下方已知問題，
   命令列的 git 沒有設定憑證，push 不了）→ Render 會自動偵測新 commit 部署，或手動 Manual Deploy

**Windows / Git Bash 已知問題**：
- 把 `postgresql://...` 這種 URI 格式連線字串當command-line參數傳給 `psql`（或任何原生 Windows exe）
  在 Git Bash 下會解析錯誤（"extra command-line argument" 警告，之後的操作會卡住/失敗）。
  解法：改用 `PGHOST`/`PGPORT`/`PGDATABASE`/`PGUSER`/`PGPASSWORD`/`PGSSLMODE` 環境變數，
  `psql`/`psycopg2.connect('')` 不帶參數直接讀環境變數
- 本機這台電腦的 git 沒有設定 GitHub 憑證（HTTPS push 會出現 "Repository not found"，即使 repo 真的存在），
  push 一律透過 **GitHub Desktop**（它有自己的登入機制）

**已知效能限制與正確性取捨（2026-07-05，加 timing log 後用 Render 實測數字更正）**：

- `/api/optimize` 系列 endpoint（`api/main.py` `_run_optimize`）耗時組成：
  - DB 查詢（`prepare_batter_balls`，cache miss 時）：約 4~10 秒
  - 單次 `optimize_positions`（`n_restarts=20`，無併發競爭）：實測約 **44~50 秒**，不是先前誤估的「約 40 秒」
  - 一般模式（未指定外野手）在有 `home_team` 時會**循序呼叫兩次** `optimize_positions`（`no_park` + `with_park`），
    單一請求總耗時可達 **2 分鐘以上**（實測 135 秒）
- **`n_restarts=20` 本身有約 12% 機率收斂到非全域最優解**（見「核心程式」章節 `optimize_positions` 條目下
  2026-07-05 的驗證數字），是目前為了免費方案速度接受的正確性折衷，不是無成本的最佳化。
- ⚠️ **CPU 併發競爭（2026-07-05 發現並修復）**：Render 免費方案只有 0.1 CPU，多個 `optimize_positions`
  同時執行會互搶 CPU、讓每一個都變慢（實測：單獨跑 44~50 秒，兩個同時跑各要 70~81 秒）。前端比較模式
  （`frontend/src/App.jsx` `Promise.all` 同時送出 A/B 兩個 `/api/optimize_plot` 請求）若雙邊都沒指定外野手，
  等於同時有最多 4 個 `optimize_positions` 搶同一個 0.1 CPU，這正是使用者曾實測到 **5 分鐘**回應時間的根因
  （不是部署過期或回歸，已用 Render deploy log 排除）。
  **已修復**：`api/main.py` 加了 `threading.Semaphore(1)`（`_optimize_semaphore`）把三個 `optimize_positions`
  呼叫點都序列化，避免併發請求互相拖慢 CPU（commit `b2d2bd5`，已實測驗證：先到的請求維持 44 秒不受影響，
  後到的請求排隊等待+執行）。
  **注意：semaphore 只解決「互相拖慢」，沒有解決「總運算量」**——比較模式最壞情況（雙邊都跑 no_park+with_park）
  仍需排隊跑滿 4 次運算，最後排到的請求總時間仍可能到 3~5 分鐘，這是結構性問題，尚未解決。
- 之後如果要同時兼顧速度與正確性，可行方向：
  1. 升級付費方案（Starter $7/月，0.5 CPU）換更多 CPU，`n_restarts` 也能調回更高（50 或 100）換取正確性
  2. **`with_park` 用 `no_park` 的解當 warm start，且把 `n_restarts` 從 20 降到 8（2026-07-05 已實作並上線，
     見上方 `optimize_positions` 條目的 30 樣本驗證數據）**：miss rate 跟原本 `n_restarts=20`（無 warm start）
     完全一樣（3/30），本機測約快 2.5 倍。**注意這個加速只套用在 `with_park` 這一次呼叫**，`no_park`（沒有
     warm start 可用）與 `custom`（指定外野手模式）仍是 `n_restarts=20`，尚未加速——如果要進一步縮短，
     這兩個路徑需要走方向 1（升級付費方案）而非 warm start（沒有前一次解可以頂替起點）
     - ⚠️ **試過但沒用、不要重複嘗試**：用 `league_avg`（聯盟平均站位）當 `no_park` 的 warm start（因為
       `no_park` 沒有前一次解可借，`league_avg` 是唯一現成的固定起點）。30 樣本驗證（`random.seed(456)`，
       對照 `n_restarts=150`）結果：`n_restarts=20` 時 miss rate 打平（5/30 vs 5/30），但 15/10/6/5 這幾個
       值反而**變差**（例如 10 時 9/30 vs 不加 warm start 的 8/30）。原因：`league_avg` 是固定的聯盟平均，
       跟特定打者的實際擊球分佈可能差很多，不是這個打者問題的好先驗，反而擠掉一個更有效的隨機起點名額。
       `with_park` 的 warm start 有效是因為跟 `no_park` 幾乎是同一個問題（只差幾顆打牆球），`league_avg`
       沒有這個「幾乎同一問題」的關係，不能類比套用。順帶測出 `no_park` 本身在 `n_restarts=20` 的 miss rate
       是 5/30（17%），比 `with_park` 的 10% 略高——no_park 的搜尋空間本身可能比較難收斂。
  3. **實際站位距離（不只目標函數值）也驗證過（2026-07-06）**：目標函數的 miss（差距通常僅 1~2%）
     背後，實際站位距離可能差到 100~200 英呎（地形在某些方向很平——多個位置對某打者幾乎一樣好）。
     30 樣本測到 `no_park` 要 `n_restarts=50` 才把最糟案例壓到 19ft、`n_restarts=75` 才近乎完全一致
     （0.1ft），`with_park` 更頑固，`n_restarts=50` 都還卡在同一個 92.8ft 的案例，要到 75 才解決。
     換算 Render，`n_restarts=75` 單次呼叫約 120~155 秒，`no_park`+`with_park` 兩次加起來會回到
     4 分鐘以上——等於把目前的加速成果賠光。
     - ⚠️ **試過但沒用的兩個演算法改動，不要重複嘗試**：
       (a) **分解成三個 2D 子問題輪流優化**（固定兩個守備員、只優化第三個，迭代直到收斂）：
       同樣耗時下（約 1.1~1.3 秒）沒有明顯贏過單純拉高 joint 6D 的 `n_restarts`，且外層起點數從
       12 加到 20，最大誤差反而從 102.7ft 變 111.0ft（不是單調變好），顯示有特定案例會讓這個方法
       結構性卡住，不是起點不夠多的問題。
       (b) **自適應提前停止**（連續 N 次沒有找到更好解就提早停止，設下限與上限）：更糟，難案例常常
       在達到最低次數（10）後就因為「連續幾次沒進步」被誤判為已收斂而提早停手，實際上還沒找到真解
       ——因為某些案例的真正全域最優解本身只佔搜尋空間很小一塊，隨機抽不到不代表已經找完，
       「沒進步」跟「已收斂」在這裡是兩回事。
     - **結論（2026-07-06）**：目前沒有找到「免費、快、又完全穩定」三者兼得的方法，這是嘗試多種
       演算法後得到的結論，不是還沒試過。**已決定維持現狀（快+免費，接受少數案例的站位不穩定）**，
       不繼續往「拉高 n_restarts 到 50~75」或「升級付費方案」的方向調整。之後如果要再處理這個問題，
       比較有希望的方向是「熱門查詢離線用高 n_restarts 預先算好存起來」（快取思路，但目的是穩定
       不是省算力）或「先回傳快速結果、背景用高 n_restarts 重算後更新畫面」，兩者都還沒實作、
       也還沒驗證是否可行（後者要處理 Render 0.1 CPU 下背景任務與即時請求搶資源的問題）。

## ⚠️ 重要陷阱與慣例
1. **OAA 必須用群體層 mu_***，不能用球員自己的隨機效應 → 否則用球員自己的資料預測自己，
   OAA 訊號自我抵銷（循環論證）。曾因此 evaluate 腳本 bug 讓 R 假性偏低（0.68），改正後回到 0.79。
2. **speed+dist+time 三角共線**：speed=dist/time，三個同時進模型時後驗 sd 膨脹、係數方向不穩定（已實測）。
   定案版本選 speed+dist（排除 time），VIF 最大 1.445，穩定。
3. Model_3 是**論文封存**，其程式與資料一律不動。
4. 欄位語意要實際印資料核對（statcast 的 player_name=投手，不是守備員）。
5. **savant 達標門檻**：每隊出賽場數 × 1 次守備機會（2025 約 165 次），使用**全部難度（含 0 星）**的 n_opp，不是只算困難球。
6. **官方 OAA 現在存在 DB**：`oaa_leaderboard` 表（2025 已載入 326 人），是 evaluate 腳本的可替代驗證來源。
7. **model_oaa 中心化（query time）**：`model_oaa` 表存原始值。`/api/fielders` 與 `/api/player_trend` 在 query time 做跨位置中心化：`centered = raw_oaa − avg_oaa_per_ball × n_opp`，其中 `avg_oaa_per_ball` 由當年所有 model player 的 raw_oaa / n_opp 算出。根本原因（賽季平均站位造成正偏）尚未解決，中心化只是去掉系統偏差讓跨位置比較有意義。
8. **`responsible` 守備員責任分配用距離，不用 `_catch_prob_single_fielder` 跨位置比較**：
   `_catch_prob_single_fielder` 的 cos/sin 特徵是「相對徑向角」（`rel = run_angle - (pos_angle + π)`），
   跨位置比較時，同顆球從不同守備員角度的 `cos(rel)` 值差異可能蓋過距離懲罰，
   導致較遠守備員機率反而更高（OOD 行為，訓練資料中守備員只追自己責任區的球）。
   目前定案：用 `np.hypot` 距離決定 `responsible`。`optimization.py` 中留有
   `compute_per_fielder_probs`（各守備員對每顆球的個別接殺機率）與
   `compute_ball_catch_probs`（三人聯合接殺機率 p̂_j）供未來研究用，兩者都是用完整模型，
   目前**沒有**只用 dist+speed 的簡化版本。
9. **合併外野手模型（OF unified）**：2026-06-27 完成，存放於 `models/2025/OF/`，LF+CF+RF 共用同一 scaler 與群體層參數（mu_alpha=-2.159, mu_beta_speed=-20.124, mu_beta_dist=+1.917）。
10. **官方球集 vs 全量差異**：Crow-Armstrong CF 2025 例，官方 455 球 vs 全量 641 球。差的 186 球全未接到，median required_speed = 39 ft/s（官方球集 = 14.4 ft/s）。Statcast 排除閾值估計約 28 ft/s（人體衝刺上限），但單純閾值無法精確復現 455，官方可能有額外條件。
11. **model_oaa 數值尺度比官方 OAA 誇張**：`validation_scatter_v2.png` 用等比例座標＋y=x 線揭露的現象——2025 樣本外 model OAA 的標準差（≈14）約為官方 OAA 標準差（≈7）的兩倍，迴歸斜率≈0.41（不是理想的 1.0）。排名相關（R=0.796）仍成立，但數值大小不能直接當官方 OAA 讀，這跟第 7 點的中心化偏正是同一個根本原因（賽季平均站位）的另一種症狀，中心化目前只修正平均值偏移，沒修正尺度（方差）。

## 特徵消融實驗結果（2026-06-26，2025 樣本外、official 子集 n=89）

| 模型 | 特徵 | R vs 官方 OAA | Brier (is_official) | VIF 最大值 |
|---|---|---|---|---|
| **speed+cos+sin+dist（定案）** | 4 項 | **0.7952** | 0.07245 | 1.445 |
| time+speed+cos+sin+time×cos（完整版） | 5 項 | 0.7869 | 0.07189 | ~12 |
| time+speed+cos+sin（無交互） | 4 項 | 0.7854 | 0.07248 | — |
| speed+cos+sin+dist+time（三角共線） | 5 項 | 0.7890 | 0.07246 | 後驗 sd 膨脹 |
| speed only | 1 項 | 0.7618 | 0.08133 | — |
| pass18_2124（論文 Model_3） | 5 項 | 0.785 | BSS=0.7507 | — |

定案模型全 2025 球校正指標（n=52,598，不限 is_official）：Brier=0.0632，LogLoss=0.2142，MAE=0.1278

選 `speed+cos+sin+dist` 的理由：R 最高、VIF 最乾淨、係數物理意義清楚（speed=難度、dist=時間充裕度）。
R 與論文 pass18_2124 的差距（+0.010）在 n=89 下不具統計顯著性，強項是結構更乾淨。
