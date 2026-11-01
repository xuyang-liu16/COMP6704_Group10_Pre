# Open (LIU Xuyang)

## Machine learning model training via convex optimization

- COMP6704, Group 10
- L2-regularized binary logistic regression
- Presenter lead: 刘旭洋 LIU Xuyang, 26123832r
- Members: 王泽卿 WANG Zeqing; 武澳奇 WU Aoqi; 赵纪鑫 ZHAO Jixin
- Same training objective; GD, Nesterov, FISTA, ADMM

### Speaker bullets: 00:00-01:00, 旭洋

- 介绍组员和问题：用固定特征预测二分类标签，训练变量是权重与偏置。
- 同一份数据、同一个目标函数，比较优化器达到指定解精度所需的计算。
- 强调范围：固定特征的逻辑回归是凸问题；结论不直接覆盖深度网络训练。
- 指向四个方法名称，停顿约 5 秒，让听众建立比较对象。

## One objective, separate optimization and prediction

$$
\min_{w\in\mathbb{R}^d}
\frac{1}{n}\sum_{i=1}^n\log(1+\exp(-y_i x_i^\top w))
+\frac{\lambda}{2}\|w\|_2^2,\qquad y_i\in\{-1,+1\}.
$$

- Bias is appended to the features and regularized.
- Training objective and gradient norm measure optimization progress.
- Full-call runtime measures computational cost.
- Held-out accuracy measures prediction.

### Speaker bullets: 01:00-02:00, 旭洋

- 指公式两部分：平均 logistic loss 对应拟合，L2 项控制参数大小。
- 提醒偏置也受惩罚，避免后面与库函数对比时混用不同目标。
- 测试集只做评价。训练目标更低，不保证测试准确率更高。
- 不提前宣告算法排名；最终结论需要共同精度下的完整实验。

## Presentation route

| Time | Speaker | Focus |
| --- | --- | --- |
| 00:00-03:00 | 旭洋 | Task and comparison questions |
| 03:00-10:00 | 泽清 | Formulation, convexity, GD |
| 10:00-12:00 | 旭洋 | Implemented Nesterov sequence |
| 12:00-19:00 | 奥奇 | FISTA and experiment design |
| 19:00-26:00 | 继昕 | ADMM and results/evidence status |
| 26:00-30:00 | 旭洋 | Conclusions and limitations |

### Speaker bullets: 02:00-03:00, 旭洋

- 先回答为什么问题是凸的，再比较更新公式，最后讨论计算与预测结果。
- 理论、实现、实验分别陈述：理论收敛率不等于实测加速倍数。
- 说明每位成员负责一个算法，旭洋组织开场、Nesterov 和总结。
- 交接原句：“下面请泽清说明变量、凸性和梯度下降基线。”

Presenter-only note: the short speaker labels follow the requested script
convention. Formal member names above follow `AGENTS.md`. Total opening
time is 3 minutes; Q&A follows the separate 30-minute presentation.
