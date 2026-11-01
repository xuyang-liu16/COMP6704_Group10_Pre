# Close (LIU Xuyang)

## Choosing a solver

| Method | Useful structure | Main cost or risk |
| --- | --- | --- |
| GD | Smooth objective; baseline implementation | Slow progress under poor conditioning |
| Nesterov | Smooth objective; look-ahead acceleration | Oscillation; step-size validity |
| FISTA | Smooth loss plus inexpensive prox | Split and momentum determine behavior |
| ADMM | Splitting, constraints, consensus | Inner solves and penalty tuning |

### Speaker bullets: 26:00-27:00, 旭洋

- GD 适合建立基线、核对目标和梯度；先保证大家解同一问题。
- 本实现 Nesterov 使用一般凸的加速序列，理论目标差为 $O(1/k^2)$。
- 不把强凸固定动量版本的线性速率写成本实现的保证。
- 默认步长下，L2-prox FISTA 与 Nesterov 更新等价，轨迹重合符合预期。

## What the evidence supports

- Same objective, data, regularization, and initialization.
- Compare full-call time to a common quality target.
- Check original-objective gradient norms and ADMM residuals.
- Keep training loss and test accuracy separate.

Single-run synthetic snapshot, seed 0, lambda 0.01, gradient tolerance
`1e-6`; source: `results/summary.json`, SHA-256
`cdce2dd2a0820bc5ede2404730a7dc9af109d9967a53139f3ea3e1f29f812487`.

| Method | Updates | Full-call seconds | Test accuracy |
| --- | ---: | ---: | ---: |
| GD | 136 | 0.007084 | 73.80% |
| Nesterov | 138 | 0.008519 | 73.80% |
| FISTA | 138 | 0.008109 | 73.80% |
| ADMM | 195 | 0.054900 | 73.80% |

### Speaker bullets: 27:00-28:00, 旭洋

- 四个方法均达到梯度阈值，最终目标约 0.493214，测试准确率相同。
- GD 在这次记录中最快；单次毫秒级计时不证明跨重复或跨数据的稳定优势。
- 仍待补充：`[TBD_TIME_VARIATION]`、跨种子统计和完整工作量。
- 若重跑后记录未核验，使用备用句：“本页保留已核验快照，新一轮排名尚待核对。”
- ADMM 的外层迭代包含内层求解，不能直接与一次梯度更新等价。

## Limits and next experiments

- Small dense datasets; no claim about neural-network training.
- Multiple seeds, repeated timing, and a reference solution are needed.
- Setup, gradient monitoring, and ADMM inner work affect runtime.
- Next: conditioning sweeps, restart, composite penalties, consensus ADMM.

### Speaker bullets: 28:00-29:00, 旭洋

- 指出标准化、正则强度会同时影响模型和条件数。
- 谱范数初始化也有成本；优先使用包含完整调用的计时。
- 下一步先补共同精度与重复运行，再扩展 L1/elastic net 和分布式场景。
- 单机 ADMM 实验不能支持分布式加速或隐私保证。

## Takeaways and discussion

- Verify the objective before comparing solvers.
- Match the method to smoothness, proximal structure, and decomposition.
- Report solution quality together with total computational cost.

### Speaker bullets: 29:00-30:00, 旭洋

- 归纳条件性建议：GD 作基线，平滑集中式训练评估 Nesterov，
  有近端结构考虑 FISTA，有分解需求考虑 ADMM。
- 用已核验结果收束：当前一次合成数据运行中解质量相近，GD 记录时间最短。
- Q&A 分工：泽清负责凸性/GD；旭洋负责 Nesterov/总体取舍；
  奥奇负责 FISTA/公平实验；继昕负责 ADMM/结果。
- 结束原句：“谢谢各位，下面进入问答。”

Presenter-only note: closing is exactly the scheduled 4-minute block.
Do not read placeholder tokens aloud. Use the fallback sentence if the
results have not been verified. The 10-minute Q&A is additional.
