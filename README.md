# D3：根据过去的家庭用电，预测下一小时平均功率

根据一户家庭的连续分钟级用电记录，比较**持续性基线**（“下一小时和最近一小时差不多”）与一个**时间序列候选模型**（随机森林），预测下一小时的平均有功功率（kW）。

- **使用者**：希望提前看到用电变化、但不能依赖昂贵系统的家庭能源研究小组。
- **重要错误**：突增/突降时刻预测误差很大，尤其是傍晚 17–20 点。
- **边界**：只做单户历史数据实验，不承诺推广到其他家庭，不用于电网控制或安全告警。

## 真实数据

- 来源：[Kaggle — Household Electric Power Consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)
- 数据文件：`data/raw/household_power_consumption.txt`（`;` 分隔，表头 9 列，数据 2,075,259 行，缺失值用 `?`）
- 许可标签：Database: Open Database, Contents
- 说明：原始大数据不提交进 Git；本仓库只保存清洗后的小时数据 `data/processed/hourly_power.csv`。

## 环境与安装

Python 3.11+，然后：

```powershell
python -m pip install -r requirements.txt
```

依赖：`numpy`、`pandas`、`scikit-learn`、`matplotlib`。

## 运行步骤

所有命令都在本目录（`student-work/day-03-power`）下运行。

### 1. 数据检查与准备

把真实分钟记录清洗并聚合成小时平均功率（默认读取前 150,000 条分钟记录）：

```powershell
python analyze.py --prepare
```

预期输出：

```
REAL DATA PREPARATION PASSED
source_rows_requested: 150000
hourly_rows: 2501
start: 2006-12-16 17:00:00
end: 2007-03-30 21:00:00
output: data\processed\hourly_power.csv
```

### 2. 主程序（基线 vs 候选）

构造滞后特征 `lag_1/2/3/24`、`hour_of_day` 与 `day_of_week`，按时间顺序 80/20 划分（不随机打乱），在同一测试段上比较持续基线（`lag_1`）与固定种子随机森林，并输出指标、对比图和最大误差：

```powershell
python analyze.py
```

预期输出（`metrics.json` 关键字段，真实数据运行结果）：

```json
"baseline":  { "mae_kw": 0.794, "rmse_kw": 1.112 },
"candidate": { "mae_kw": 0.541, "rmse_kw": 0.759 }
```

### 3. 测试

```powershell
python -m unittest discover -s tests -v
```

预期：`Ran 2 tests ... OK`（检查时间顺序划分保持有序、滞后特征只用过去值预测下一小时）。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `data/processed/hourly_power.csv` | 小时平均功率（2,501 行） |
| `metrics.json` | 基线与候选的 MAE/RMSE、高需求告警统计 |
| `forecast.png` | 第一个留出周：实际 vs 基线 vs 候选对比图 |
| `largest_errors.csv` | 候选模型最大误差 12 条（时间、真实值、预测、误差） |
| `report.md` | 书面报告 |
| `presentation.pptx` | 3 分钟答辩 |

## 结果摘要（真实数据）

- 数据窗口：2006-12-16 至 2007-03-30，训练 1,980 小时 / 测试 496 小时。
- 基线 MAE **0.794 kW**；候选 MAE **0.541 kW**（低约 32%）。
- 高需求（≥3.16 kW）误报：基线 31 次 → 候选 9 次；召回：0.184 → **0.237**（TP 7 → 9）。
- 最大失败案例：`2007-03-17 22:00`，真实 0.30 kW、预测 3.31 kW、误差 3.02 kW（傍晚高峰后入睡骤降）。

## 方法与限制

- **基线**：最近一小时功率（`lag_1`），零参数、可手算。
- **候选**：随机森林（200 棵树、深度 15、叶子最小 2、`random_state=42`），特征只用预测时刻之前的 `lag_1/2/3/24`、整点小时数与星期几，不使用任何未来信息。深度/叶子比 starter 默认略低正则化，并加入 `day_of_week`，以提高高需求召回率。
- **比较公平性**：基线与候选使用同一小时数据、同一时间顺序划分、同一 MAE/RMSE/高需求指标。
- **限制**：单一家庭、单一约 3.5 个月时段；滞后特征无法刻画无规律的突发用电（最大误差集中在傍晚突变时刻）；结论不推广到其他家庭，不用于电网控制或安全告警。

## 已知说明

课程文档中提到的 `python analyze.py --check-data` 与实际程序参数不一致：本程序实际使用 `python analyze.py --prepare`（输出 `REAL DATA PREPARATION PASSED`），报告按真实程序记录。
