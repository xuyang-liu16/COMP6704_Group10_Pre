<div align="center">
<h1>COMP6704 Group 10 口头汇报</h1>
<p><b>机器学习中的凸优化</b><br/>
L2 正则逻辑回归</p>
<p>
  <a href="README.md">English</a> | <strong>中文</strong>
</p>
<p>
刘旭洋 LIU Xuyang · 王泽卿 WANG Zeqing · 武澳奇 WU Aoqi · 赵纪鑫 ZHAO Jixin<br/>
香港理工大学 电子计算学系<br/>
COMP6704 · 口头报告 2026年11月3日 · 30 分钟 + 10 分钟问答
</p>
</div>

我们用 **四种** 凸优化求解器训练 **同一个** L2 正则逻辑回归模型。每位成员主讲一种方法。

| 方法 | 主讲 | 代码 |
| --- | --- | --- |
| 梯度下降 (GD) | 王泽卿 WANG Zeqing | [`algorithms/gd.py`](algorithms/gd.py) |
| Nesterov 加速梯度 | 刘旭洋 LIU Xuyang | [`algorithms/nesterov.py`](algorithms/nesterov.py) |
| FISTA（近端梯度） | 武澳奇 WU Aoqi | [`algorithms/fista.py`](algorithms/fista.py) |
| ADMM | 赵纪鑫 ZHAO Jixin | [`algorithms/admm.py`](algorithms/admm.py) |

## 📝 目录

- [问题设定](#-问题设定)
- [算法文件位置](#-算法文件位置)
- [仓库结构](#-仓库结构)
- [快速开始](#-快速开始)
- [口头汇报幻灯片](#-口头汇报幻灯片)
- [汇报分工与时间](#-汇报分工与时间)
- [成员](#-成员)

## 🎯 问题设定

最小化平均逻辑损失加上 L2 惩罚项：

\[
\min_w \; F(w) = f(w) + g(w), \quad
f(w)=\tfrac{1}{n}\sum_i \log\bigl(1+e^{-y_i x_i^\top w}\bigr),\quad
g(w)=\tfrac{\lambda}{2}\|w\|_2^2
\]

其中 \(\lambda=0.01\)。共享工具在 [`problem.py`](problem.py)、[`data.py`](data.py)、[`metrics.py`](metrics.py)。

## 📂 算法文件位置

| 文件 | 方法 | 负责人 | 实现要点 |
| --- | --- | --- | --- |
| [`algorithms/gd.py`](algorithms/gd.py) | 梯度下降 | 王泽卿 | 全批量 GD + Armijo 回退；步长与 \(1/L\) 相关 |
| [`algorithms/nesterov.py`](algorithms/nesterov.py) | Nesterov | 刘旭洋 | 前瞻点动量；一般凸 \(t\) 日程；在 \(\|\nabla F(w)\|\) 上停机 |
| [`algorithms/fista.py`](algorithms/fista.py) | FISTA | 武澳奇 | 只对损失求梯度 + L2 近端闭式收缩；动量与 Nesterov 同构 |
| [`algorithms/admm.py`](algorithms/admm.py) | ADMM | 赵纪鑫 | 拆分 \(w/z\)、对偶更新、残差检验与内层求解 |

**不要跨改**其他成员的算法文件。分工见 [`AGENTS.md`](AGENTS.md)。

对比实验与图表：

| 路径 | 作用 |
| --- | --- |
| [`experiments/run_compare.py`](experiments/run_compare.py) | 在合成 / 乳腺癌数据上跑四种求解器 |
| [`scripts/make_figures.py`](scripts/make_figures.py) | 绘制目标函数–迭代 / 时间曲线 |
| [`results/`](results/) | 已保存的 JSON 表与曲线（勿编造数字） |
| [`figures/`](figures/) | 幻灯片/报告用图 |

## 🗂 仓库结构

```text
.
├── algorithms/          # 四种求解器（各一个文件）
├── experiments/         # 对比实验入口
├── figures/             # 曲线图
├── results/             # summary.json、curves.json、表格
├── report/sections/     # 报告各节（按负责人）
├── slides/
│   ├── Group10_Oral.pptx              # 可编辑口头 PPT（已嵌入 Latin Modern）
│   ├── Group10_Oral_PolyU_Beamer.pdf  # Beamer 视觉参考
│   ├── Group10_Oral_PolyU_Beamer.tex
│   └── beamer/Group10/                # Beamer 源码
├── SPEAKING_SCRIPTS_EN.md             # 英文口播 + 计时
├── SPEAKING_SCRIPTS_BILINGUAL.md      # 英在前、中文在后
├── Group10_Oral.pptx                  # 与 slides/ 下 PPTX 同字节副本
└── requirements.txt
```

## 🚀 快速开始

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 四种求解器对比（结果写入 results/）
python experiments/run_compare.py

# 重新导出图
python scripts/make_figures.py
```

## 🎤 口头汇报幻灯片

- 可编辑 PPTX：[`slides/Group10_Oral.pptx`](slides/Group10_Oral.pptx)（32 页可见 + 2 页隐藏备份；Latin Modern 已嵌入，方便 WPS 打开）
- 交付说明：[`slides/PPTX_DELIVERY.md`](slides/PPTX_DELIVERY.md)
- Beamer PDF/TeX：[`slides/Group10_Oral_PolyU_Beamer.pdf`](slides/Group10_Oral_PolyU_Beamer.pdf)

## ⏱ 汇报分工与时间

| 环节 | 主讲 | 页码 | 时钟 |
| --- | --- | ---: | --- |
| 开场 | 刘旭洋 | 1 | 00:00–00:25 |
| 背景 + 共享问题设定 | 王泽卿 | 2–6 | 00:25–04:20 |
| 四方法总览 | 刘旭洋 | 7 | 04:20–05:00 |
| 梯度下降 GD | 王泽卿 | 8–11 | 05:00–08:00 |
| Nesterov | 刘旭洋 | 12–15 | 08:00–12:00 |
| FISTA + 与 Nesterov 等价 | 武澳奇 | 16–19 | 12:00–16:00 |
| ADMM | 赵纪鑫 | 20–23 | 16:00–20:00 |
| 实验对比与结论 | 赵纪鑫 | 24–31 | 20:00–29:00 |
| 收尾 | 刘旭洋 | 32 | 29:00–30:00 |

完整口播：[`SPEAKING_SCRIPTS_EN.md`](SPEAKING_SCRIPTS_EN.md) · [`SPEAKING_SCRIPTS_BILINGUAL.md`](SPEAKING_SCRIPTS_BILINGUAL.md)

## 👥 成员

| 姓名 | NetID | 负责 |
| --- | --- | --- |
| 刘旭洋 LIU Xuyang | 26123832r | `algorithms/nesterov.py`，开场/收尾，formulation/conclusion |
| 王泽卿 WANG Zeqing | 26029011r | `algorithms/gd.py`，共享设定，introduction |
| 武澳奇 WU Aoqi | 26027031r | `algorithms/fista.py`，methods，实验辅助代码 |
| 赵纪鑫 ZHAO Jixin | 26009105r | `algorithms/admm.py`，results，实验对比页 |

## 说明

本仓库为香港理工大学 COMP6704 课程作业材料，仅供课程教学与演示使用。
