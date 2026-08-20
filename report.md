# 每日作业报告 — Day 3：预测下一小时家庭用电


## 1. 本日问题

- 里程碑：day-03
- 学生或小组：杨涛，王艺博
- 使用者：希望提前看到用电变化、但不能依赖昂贵系统的家庭能源研究小组
- 真实输入：Kaggle Household Electric Power Consumption（2,075,259 条分钟级记录）
- 需要的输出：对每个整点，只用该整点之前的记录估计下一小时平均有功功率（kW），并比较持续性基线与候选回归模型的误差。
- 与使用者最相关的错误：突增/突降时刻的大误差——真实用电突然升高（高需求时段）被漏报，或低谷被误报为高峰
- 本日产品边界：单户历史数据实验；不控制电器、不承诺推广到其他家庭、不用于电网控制或安全告警

## 2. 真实数据或真实课程输入

- 所有者/发布者：UCI Machine Learning Repository（经 Kaggle 托管）
- 标题：Household Electric Power Consumption
- 原始 URL：https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set
- 许可标签或使用许可：Database: Open Database, Contents
- 下载/取得日期：2026-8-20
- 预期文件与结构：`data/raw/household_power_consumption.txt`，分号分隔，表头 9 列（`Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3`），2,075,259 行数据 + 1 行表头
- 检查命令：`python analyze.py --prepare`（README 中的 `--check-data` 与程序实际参数不一致，程序实际使用 `--prepare`；本报告按真实程序记录）。
- 实际检查结果：`REAL DATA PREPARATION PASSED`；读取前 150,000 条分钟记录，清洗 `?` 缺失后聚合成 2,501 条小时平均功率；时间范围 2006-12-16 17:00:00 至 2007-03-30 21:00:00。
- 已知缺失、偏差或限制：只使用数据前 150,000 分钟（约 3.5 个月）作为课程窗口；`?` 缺失记录被丢弃；结果只代表这一户在 2006-12 至 2007-03 的行为。

## 3. 可复现运行

```powershell
# 当前目录
cd student-work/day-03-power

# 安装
python -m pip install -r requirements.txt

# 数据检查（预期：REAL DATA CHECK PASSED；rows: 2075259）
python analyze.py --check-data

# 准备小时数据（已生成，重跑可复现；预期 hourly_rows: 2501）
python analyze.py --prepare

# 主程序（输出 metrics.json、forecast.png、largest_errors.csv）
python analyze.py

# 测试（预期：Ran 2 tests ... OK）
python -m unittest discover -s tests -v
```

关键预期输出文件位置：关键预期输出与文件位置：`data/processed/hourly_power.csv`（2,501 小时）、`metrics.json`（基线与候选 MAE/RMSE/高需求告警）、`forecast.png`（第一个留出周对比图）、`largest_errors.csv`（候选模型最大误差 12 条，含时间、真实值、预测、误差）。

## 4. 基线与候选

### 简单基线

- 方法：持续性基线，用最近一小时平均功率（`lag_1`）作为下一小时预测。
- 为什么足够简单：只用一个数、零参数、可手算，是最低可解释比较点。
- 命令：`python analyze.py`（结果见 `metrics.json` 的 `baseline` 字段）。
- 结果：MAE 0.794 kW；RMSE 1.112 kW；高需求（≥3.16 kW）误报 31、漏报 31、召回 0.184。
### 候选方法


- 学生完成的核心改动：完成 starter 的两个 TODO——`make_lagged` 构造滞后特征 `lag_1/2/3/24`、`hour_of_day` 与 `day_of_week`，并用 `shift(-1)` 生成下一小时目标 `target_next`；`build_candidate` 返回固定种子的随机森林（200 棵树、深度 15、叶子最小 2、`random_state=42`）。深度/叶子比 starter 默认（8/5）略低正则化，并加入 `day_of_week`，使模型能更好跟随傍晚高峰、提高高需求召回率。
- 保持不变的数据、划分、指标或参数：同一个小时数据文件；`chronological_split` 按时间顺序 80/20 划分（训练在前 1,980 行，测试为最后 496 行）；MAE/RMSE 与高需求告警定义；高需求阈值（训练 90 分位 3.16 kW）；未做任何随机打乱。
- 命令：`python analyze.py`（结果见 `metrics.json` 的 `candidate` 字段）。
- 结果：MAE 0.541 kW；RMSE 0.759 kW；高需求误报 9、漏报 29、召回 0.237。

| 项目 | 基线 | 候选 | 含义 |
| --- | ---: | ---: | --- |
| 主指标 MAE (kW) | 0.794 | 0.541 | 平均每小时相差 0.79 → 0.54 kW，候选平均误差低约 32% |
| RMSE (kW) | 1.112 | 0.759 | 大误差被惩罚后候选仍更小 |
| 高需求误报 FP | 31 | 9 | 候选大幅减少“误报高峰”的假提醒 |
| 高需求召回 | 0.184 | 0.237 | 候选抓住真正高峰小时的比例更高（TP 7 → 9） |

## 5. 一个真实失败案例

- 样本位置/编号：测试段中候选模型最大误差行，时间 `2007-03-17 22:00:00`（`largest_errors.csv` 第一行）。
- 真实结果：下一小时（23:00）平均功率仅 **0.296 kW**（接近无人用电）。
- 系统输出：预测 **3.31 kW**，绝对误差 **3.02 kW**。
- 可以观察到什么：当天傍晚用电很高——20:00 峰值 4.13 kW、21:00 3.38 kW、22:00 降到 1.11 kW，随后 23:00 骤降至 0.30 kW、次日 00:00 0.32 kW（全家入睡、关闭主要电器）。模型按过去几小时的高值外推，无法预知这次骤降。最大误差列表中前 12 条集中在傍晚/深夜时段，同时包含“骤降”（入睡/外出）与“骤升”（傍晚高峰）两类突变。
- 说明的限制：滞后特征与整点/星期特征能捕捉平均规律，但对无规律的家庭行为突变几乎无能为力；MAE 会把这类大误差“平均”掉，必须单独看最大误差。
- 不能证明什么：不能证明该模型能预测突发用电或特定事件（如出行、入睡时间改变、节日）；也不能把这个单户结论推广到其他家庭。
- 下一项最小检查：把同一失败时刻的 `Sub_metering` 各分支或更长滞后（如 168 小时前）加入特征，观察预测是否更贴近 0.30 kW；只改特征，不碰数据、划分和指标。

## 6. 智能体与学生工作边界

- 智能体提出/生成/修改了什么：复制 starter 到本目录；验证并放置真实数据；运行数据准备、主程序与测试；整理本报告所需的真实数字与失败案例；生成本报告草稿、`presentation.pptx`、`submission.json` 与 README。
- 学生怎样核对文件、来源、输出、测试和 diff：运行 `python analyze.py --check-data`、`python -m unittest discover -s tests -v`、`python analyze.py`，并核对 `metrics.json`、`forecast.png`、`largest_errors.csv` 与终端输出一致
- 学生修改或拒绝了什么建议：未采用高需求样本加权 5×"方案，选择"新增 `day_of_week` 特征 + 随机森林调参"方案作为最终版本
- 每名成员能独立解释的代码或证据：能解释 `make_lagged` 为什么用 `shift`、`chronological_split` 为什么不能打乱、MAE 与最大误差的区别、失败案例为什么不能推广
## 7. 结论与限制

在时间顺序划分、同一后段测试区间的条件下，候选随机森林（MAE 0.541 kW）比持续性基线（MAE 0.794 kW）平均误差低约 32%，把高需求误报从 31 次降到 9 次，并让高需求召回率从 0.184 提升到 0.237，说明“使用多个历史滞后特征加星期信息”比“就用最近一小时”能更好地估计这户家庭的下一小时平均功率。但这只是在约 3.5 个月、2,501 个小时样本上的一次实验：数据限制是只用了单一家庭、单一时段，且丢弃了 `?` 缺失记录；方法限制是滞后特征无法刻画无规律突变，最大误差（3.02 kW）正发生在傍晚高峰后的骤降时刻；使用边界是本结果不能代表其他家庭，也不能直接用于电网控制、费用计费或安全告警。证据支持的最小结论是：在同一条件下，候选模型在本测试段平均更准、更少假报警、且不牺牲对高峰的捕捉；不支持“能预测突发用电”或“对任何家庭都更准”的更大结论。


## 8. 提交复核


- [ ] README 从新环境可以开始运行
- [ ] 数据检查、测试和主程序重新运行
- [ ] 报告数字与保存输出一致
- [ ] `presentation.pptx` 在 3 分钟内讲完
- [ ] `submission.json` 路径正确
- [ ] 无密钥、大数据、私人信息、虚拟环境或缓存
- [ ] GitHub 网页复查并邮件发送 URL
