# COMP6704 Group 10: Bilingual speaking scripts

32 visible slides, 2 hidden Q&A backups. Talk: 30 minutes, followed by 10 minutes of Q&A.

| Part | Speaker | Slides | Duration | Clock |
| --- | --- | --- | --- | --- |
| Opening | LIU Xuyang | 1 | 0m 25s | 00:00-00:25 |
| Background and shared formulation | WANG Zeqing | 2-6 | 3m 55s | 00:25-04:20 |
| Four-method overview | LIU Xuyang | 7 | 0m 40s | 04:20-05:00 |
| Gradient Descent | WANG Zeqing | 8-11 | 3m 0s | 05:00-08:00 |
| Nesterov | LIU Xuyang | 12-15 | 4m 0s | 08:00-12:00 |
| FISTA and equivalence | WU Aoqi | 16-19 | 4m 0s | 12:00-16:00 |
| ADMM | ZHAO Jixin | 20-23 | 4m 0s | 16:00-20:00 |
| Experiments and comparison | ZHAO Jixin | 24-31 | 9m 0s | 20:00-29:00 |
| Closing | LIU Xuyang | 32 | 1m 0s | 29:00-30:00 |

Each member presents one solver. WANG owns the shared formulation, WU owns the Nesterov-FISTA equivalence, and ZHAO owns all experiments and limitations. LIU gives the opening, overview and closing.

Method handoffs: 08:00, 12:00, 16:00. Experiments begin at 20:00. Closing begins at 29:00. Bracketed cues are actions, not spoken words. Hidden slides 33-34 carry no planned talk time.

### Slide 1: Convex Optimization for Machine Learning

[Speaker: LIU Xuyang | Opening | 00:00-00:25 | 25s]

We use four methods, GD, Nesterov, FISTA and ADMM, to train the same model. Each of us presents one.

Good evening. We are Group 10. I am LIU Xuyang, with WANG Zeqing, WU Aoqi and ZHAO Jixin. Our model is L2-regularized logistic regression. Zeqing will introduce the classification problem and our shared setup.

中文：
我们使用 GD、Nesterov、FISTA 和 ADMM 四种方法训练同一个模型，每位组员讲解一种。

大家晚上好，我们是 Group 10。我是 LIU Xuyang，组员是 WANG Zeqing、WU Aoqi 和 ZHAO Jixin。模型是带 L2 正则的逻辑回归。下面由 Zeqing 介绍分类问题和共同设定。

### Slide 2: Binary classification with L2 regularization

[Speaker: WANG Zeqing | Background and shared formulation | 00:25-01:00 | 35s]

I am WANG Zeqing. Our task is binary classification: use measured features to predict one of two labels. Logistic regression assigns a linear score and converts it to a probability. We learn the weights by minimizing average logistic loss with an L2 penalty. The model stays fixed throughout this comparison. Only the solver changes. We measure both training precision and accuracy on separate test data.

中文：
我是 WANG Zeqing。任务是二分类：根据测量到的特征预测两个标签之一。逻辑回归先计算线性分数，再将其转为概率。我们通过最小化平均逻辑损失与 L2 正则项之和学习权重。比较过程中模型固定，只改变求解器。我们同时检查训练精度和独立测试数据上的准确率。

### Slide 3: One objective for four solvers

[Speaker: WANG Zeqing | Background and shared formulation | 01:00-01:50 | 50s]

This is the objective that all four methods share.

Capital F is the average logistic loss f plus the L2 penalty g. Each input x has a label y equal to minus one or plus one. The score is x transpose w. A score that supports the correct label gives a smaller logistic loss.

[Point to the two terms.]

The penalty is lambda over two times the squared weight norm. We fix lambda at zero point zero one. The bias is an extra coordinate, and we penalize it too.

Every solver uses this exact objective, so we can compare their progress toward the same answer.

中文：
这是四种方法共同使用的目标函数。

大写 F 是平均逻辑损失 f 加上 L2 正则项 g。每个输入 x 对应一个取负一或正一的标签 y。模型分数是 x 转置乘 w。分数越支持正确标签，逻辑损失就越小。

[指向这两项。]

正则项是 lambda 除以二，再乘权重范数的平方。我们固定 lambda 为 0.01。偏置作为额外的一维权重，同样接受正则化。

每个求解器使用完全相同的目标函数，因此我们能比较它们接近同一个答案的过程。

### Slide 4: Predictions and the full gradient

[Speaker: WANG Zeqing | Background and shared formulation | 01:50-02:40 | 50s]

The sigmoid turns each score into a probability p. For this formula, b rewrites the label as zero or one.

[Point to p minus b, then X transpose.]

The difference p minus b measures the prediction error in probability. Multiplying by X transpose combines the examples, and dividing by n takes the average. Adding lambda w gives the gradient of the complete objective.

GD moves against this vector. Nesterov also differentiates the full objective, while FISTA handles the penalty separately.

For class prediction, a nonnegative score means plus one. One dense gradient costs order n times d. We optimize the training objective and evaluate accuracy on separate test data.

中文：
sigmoid 将每个分数转成概率 p。在这个公式里，b 将标签改写成零或一。

[指向 p 减 b，再指向 X 的转置。]

p 减 b 表示概率预测与标签的差异。乘以 X 的转置汇总各样本，再除以 n 取平均。加上 lambda w 后，就得到完整目标函数的梯度。

GD 沿这个向量的反方向移动。Nesterov 也对完整目标求梯度，FISTA 则单独处理正则项。

预测类别时，非负分数对应正一。一次稠密梯度计算需要 n 乘 d 量级的工作。我们优化训练目标，并在独立测试数据上评估准确率。

### Slide 5: Strong convexity and a unique minimizer

[Speaker: WANG Zeqing | Background and shared formulation | 02:40-03:30 | 50s]

The Hessian H describes how the gradient changes. Logistic curvature lies between zero and one quarter for each example.

[Point to the lower and upper matrix bounds.]

Adding lambda I gives positive curvature in every direction, including the bias. The objective is strongly convex with mu equal to lambda, so it has a unique minimizer.

The upper bound L limits how quickly the full gradient changes. We obtain it by adding lambda to the loss-only bound L f.

GD and Nesterov use one over L. FISTA uses one over L f before its separate penalty step. These bounds therefore give the four implementations a common mathematical starting point.

中文：
Hessian 矩阵 H 描述梯度如何变化。每个样本对应的逻辑损失曲率都在零与四分之一之间。

[指向矩阵的上下界。]

加上 lambda I 后，每个方向都有正曲率，包括偏置方向。因此目标函数是强凸的，mu 等于 lambda，并且存在唯一最优解。

上界 L 限制完整梯度变化的速度。它等于仅针对损失部分的上界 L f，再加上 lambda。

GD 和 Nesterov 使用一除以 L。FISTA 先使用一除以 L f，再单独处理正则项。这些界为各实现提供了共同的数学基础。

### Slide 6: A common convergence certificate

[Speaker: WANG Zeqing | Background and shared formulation | 03:30-04:20 | 50s]

Our common stopping test uses the norm of the full gradient at the returned weights. The target is ten to the minus six.

Strong convexity bounds the objective gap by the squared gradient norm divided by twice lambda. Passing our test therefore certifies an objective gap at most five times ten to the minus eleven.

[Point to the certificate, then the update budget.]

We cap every run at four hundred updates. A run can exhaust that budget while already making useful predictions. Equal test accuracy alone does not certify convergence.

ADMM adds residual and inner-solve checks, which Jixin will explain. Xuyang will give the four-method overview before I return to gradient descent.

中文：
共同的停止规则检查返回权重处的完整梯度范数，目标是十的负六次方。

强凸性保证目标值差距不超过梯度范数的平方除以二倍 lambda。因此，通过检查就能保证目标值差距最多为五乘十的负十一次方。

[指向精度保证，再指向更新预算。]

所有运行最多允许四百次更新。预算耗尽时，模型可能已经能给出有用的预测。仅有相同的测试准确率并不能证明收敛。

ADMM 还要求残差和内部求解通过检查，Jixin 稍后会解释。下面由 Xuyang 概览四种方法，然后我回来讲梯度下降。

### Slide 7: Contents: four methods, four speakers

[Speaker: LIU Xuyang | Four-method overview | 04:20-05:00 | 40s]

[Point to the four rows.]

Each member presents one solver. Zeqing presents GD, and I present Nesterov. Aoqi presents FISTA and explains its equivalence to Nesterov under our settings. Jixin presents ADMM and all experimental comparisons. I return for the closing. Each algorithm has four slides. We keep the same model, starting weights and convergence target throughout. Zeqing, please continue with GD.

中文：
[依次指向四行。]

每位组员讲解一种求解器。Zeqing 讲 GD，我讲 Nesterov。Aoqi 讲 FISTA，并解释它在当前设置下与 Nesterov 的等价关系。Jixin 讲 ADMM 和全部实验比较，最后由我收尾。每种算法占四页，全程使用相同模型、初始权重和收敛目标。下面请 Zeqing 继续讲 GD。

### Slide 8: GD: a full-gradient baseline

[Speaker: WANG Zeqing | Gradient Descent | 05:00-05:45 | 45s]

My solver is full-batch gradient descent. It starts with zero weights and subtracts a scaled full gradient.

The input is the same standardized training data and lambda used by the other solvers. We initialize the weights at zero.

At each accepted point, we check the full gradient norm. If the target has not been met, we form a candidate using the negative gradient. Backtracking can require several objective evaluations.

中文：
我的求解器是全批量梯度下降。它从全零权重开始，减去乘以步长的完整梯度。

输入是与其他求解器相同的标准化训练数据和 lambda。权重初始化为零。

在每个接受的点上，我们检查完整梯度范数。如果尚未达到目标，就沿负梯度方向构造候选点。回溯可能需要多次目标函数计算。

### Slide 9: GD: update and backtracking

[Speaker: WANG Zeqing | Gradient Descent | 05:45-06:30 | 45s]

[Point to the update and its step size.]

The initial trial step is one over L. The implementation evaluates the candidate objective and applies the Armijo condition shown here. The required decrease depends on the step size and the squared gradient norm.

If the candidate fails, the code halves the step and tries again. It accepts a weight vector only after that decrease check passes. The counter records accepted updates, so trial evaluations add work without adding an iteration.

For an L-smooth objective, the standard one-over-L step supplies sufficient decrease. Backtracking also protects the implementation when a caller supplies a different trial step.

中文：
[指向更新公式和步长。]

初始试探步长是一除以 L。实现会计算候选点的目标值，并检查这里的 Armijo 条件。要求的下降量取决于步长和梯度范数的平方。

如果候选点没有通过检查，代码就将步长减半后重试。只有下降条件通过，才接受新的权重。计数器只记录接受的更新，因此试探计算增加工作量，却不会增加迭代次数。

对于梯度为 L-Lipschitz 的目标函数，标准的一除以 L 步长能够满足充分下降。调用者提供其他试探步长时，回溯也能保护实现。

### Slide 10: GD: convergence rate

[Speaker: WANG Zeqing | Gradient Descent | 06:30-07:15 | 45s]

[Point to the rate bound.]

With this standard step and mu-strong convexity, the objective-error bound contracts by one minus mu over L at each update. If L is large relative to mu, that factor is close to one and the guarantee can be slow.

This gives us a concrete baseline: a simple update, a descent check, and a rate tied to the curvature bounds.

中文：
[指向收敛速度界。]

在该标准步长和 mu 强凸条件下，目标误差上界每次按一减 mu 除以 L 的比例收缩。如果 L 相对 mu 很大，这个比例就接近一，理论保证的下降可能较慢。

因此，GD 提供了明确的基线：简单更新、下降检查，以及由曲率界决定的收敛速度。

### Slide 11: GD: implementation and work

[Speaker: WANG Zeqing | Gradient Descent | 07:15-08:00 | 45s]

After acceptance, the history records the original objective and cumulative time. It includes initialization, so a history with one hundred thirty-seven points represents one hundred thirty-six updates.

The main dense work is order n times d, plus any trial evaluations and shared setup. GD has no momentum state and needs no inner matrix solve. That makes its behavior easy to audit.

Its limitation is the conservative step set by the global curvature bound. LIU Xuyang will now show how Nesterov uses previous movement to build a look-ahead point.

[Hand over to LIU Xuyang at 08:00.]

中文：
接受更新后，历史记录保存原始目标值和累计时间。记录包含初始化，因此一百三十七个历史点对应一百三十六次更新。

主要的稠密运算需要 n 乘 d 量级的工作，另外还有试探计算和共享的设置成本。GD 没有动量状态，也不需要内部矩阵求解，便于核查其行为。

它的局限是步长受全局曲率界约束，可能比较保守。接下来 LIU Xuyang 会说明 Nesterov 如何利用前一次移动构造前瞻点。

[在 08:00 交给 LIU Xuyang。]

### Slide 12: Nesterov: a look-ahead point

[Speaker: LIU Xuyang | Nesterov | 08:00-09:00 | 60s]

Thank you. I am LIU Xuyang, and my solver is Nesterov accelerated gradient.

GD evaluates its gradient at the current weights. Nesterov keeps a second vector, v, which looks ahead using recent movement. The gradient step starts from that look-ahead point.

At initialization, both vectors are zero. The first momentum coefficient is zero, so the algorithm starts with an ordinary gradient step. Later coefficients retain more of the recent direction.

[Point to the current weights and the look-ahead point. The displacement uses accepted weights from the two most recent steps.]

中文：
谢谢。我是 LIU Xuyang，我的求解器是 Nesterov 加速梯度法。

GD 在当前权重处计算梯度。Nesterov 额外保留一个向量 v，利用最近的移动向前看。梯度更新从这个前瞻点出发。

初始化时，两个向量都为零。第一次动量系数为零，所以算法从普通梯度步开始。后续系数保留更多最近的移动方向。

[指向当前权重和前瞻点。位移由最近两次接受的权重计算。]

### Slide 13: Nesterov: the accelerated update

[Speaker: LIU Xuyang | Nesterov | 09:00-10:00 | 60s]

[Point to the first equation, then the momentum equations.]

The scalar t starts at one. The second equation updates it, and the ratio in the final line controls the momentum. We add a multiple of the latest displacement to the accepted weights to form the next v.

Our implementation uses the general-convex schedule shown here. We keep the step at one over L and use neither adaptive restart nor a schedule tuned to strong convexity.

中文：
[指向第一条公式，再指向动量公式。]

标量 t 从一开始。中间的公式更新 t，最后一行中的比值控制动量。我们把最近一次位移的倍数加到接受的权重上，得到下一个 v。

实现采用这里的一般凸问题动量安排。步长固定为一除以 L，既没有自适应重启，也没有采用针对强凸性调节的安排。

### Slide 14: Nesterov: rate and stopping

[Speaker: LIU Xuyang | Nesterov | 10:00-11:00 | 60s]

The familiar one-over-k-squared objective-error rate applies to this schedule. An individual step can still increase the objective because the look-ahead point may overshoot.

First, the theoretical rate concerns the objective gap. Our experiment stops on the full-gradient norm. A faster objective-error bound does not directly determine how many updates are needed to cross a particular gradient threshold.

Third, the basic momentum schedule can oscillate. The recorded objective need not decrease at every update. We therefore retain the common gradient test instead of stopping merely because a plotted curve looks flat.

中文：
熟悉的一除以 k 平方的目标误差速度适用于这一安排。不过，前瞻点可能越过合适的位置，因此单次更新仍可能增大目标值。

第一，理论速度描述目标值差距，实验却根据完整梯度范数停止。更快的目标误差上界，并不能直接确定达到某个梯度阈值所需的更新次数。

第三，基本动量安排可能产生振荡，记录的目标值不必每步下降。因此，我们保留共同的梯度检查，不会仅凭曲线看起来变平就停止。

### Slide 15: Nesterov: work and trade-offs

[Speaker: LIU Xuyang | Nesterov | 11:00-12:00 | 60s]

[Point to the update and monitoring rows.]

The update gradient is evaluated at v, while the stopping gradient is evaluated at the accepted w. Our implementation computes both. Each dense evaluation costs order n times d, so the asymptotic order stays the same but the constant work differs from a single-gradient update.

These details help interpret the synthetic result later: GD takes one hundred thirty-six updates and Nesterov takes one hundred thirty-eight. That observation is consistent with the different stopping metric and this particular momentum schedule.

We have used the full objective in one gradient. WU Aoqi will now present FISTA, which keeps the same momentum schedule and handles the penalty through a proximal step.

[Hand over to WU Aoqi at 12:00.]

中文：
[指向更新与监控两行。]

更新用的梯度在 v 处计算，停止用的梯度在接受的 w 处计算。实现会计算两者。每次稠密计算都需要 n 乘 d 量级的工作，因此渐近量级不变，但常数成本与只计算一次梯度的更新不同。

这些细节有助于理解后面的合成数据结果：GD 用一百三十六次更新，Nesterov 用一百三十八次。这与停止指标的差异和当前动量安排并不矛盾。

我们刚才在一次梯度中处理完整目标。接下来 WU Aoqi 讲 FISTA，它保留同样的动量安排，通过近端步骤处理正则项。

[在 12:00 交给 WU Aoqi。]

### Slide 16: FISTA: separate loss and penalty

[Speaker: WU Aoqi | FISTA and equivalence | 12:00-12:40 | 40s]

I am WU Aoqi, and I present FISTA, the Fast Iterative Shrinkage-Thresholding Algorithm.

The objective still equals f plus g. We differentiate only the logistic loss f, then apply the L2 penalty g through a proximal operation.

We use eta equal to one over L f, the smoothness bound for the loss alone. The penalty enters only through this shrink step. Including lambda v in the gradient as well would count it twice.

[Point to the two terms. The loss gradient and the penalty enter through different operations.]

中文：
我是 WU Aoqi，负责 FISTA，也就是快速迭代收缩阈值算法。

目标仍然是 f 加 g。我们只对逻辑损失 f 求梯度，然后通过近端操作施加 L2 正则项 g。

我们令 eta 等于一除以 L f，使用仅针对损失的光滑性上界。正则项只通过收缩步骤进入。如果梯度中还加入 lambda v，就会重复计算正则化。

[指向两项。损失梯度与正则项通过不同的运算进入更新。]

### Slide 17: FISTA: the L2 proximal step

[Speaker: WU Aoqi | FISTA and equivalence | 12:40-13:30 | 50s]

[Point to the proximal minimization.]

The proximal problem balances the penalty against the squared distance from a trial point a. For our quadratic penalty, its derivative is lambda w plus w minus a divided by eta. Setting that expression to zero gives w equal to a divided by one plus eta lambda.

Now let a be the look-ahead point minus eta times the loss gradient. Substitution gives the update shown below. The denominator exceeds one, so every coordinate shrinks by the same factor.

中文：
[指向近端最小化问题。]

近端问题在正则项与到试探点 a 的平方距离之间取得平衡。对于二次正则项，其导数是 lambda w 加上 w 减 a 再除以 eta。令它等于零，就得到 w 等于 a 除以一加 eta lambda。

现在令 a 等于前瞻点减去 eta 乘损失梯度，代入后就得到下方更新。分母大于一，所以所有坐标都按相同比例收缩。

### Slide 18: FISTA: acceleration and work

[Speaker: WU Aoqi | FISTA and equivalence | 13:30-14:20 | 50s]

FISTA uses the same t recurrence and extrapolation as Xuyang's implementation. Its composite objective-error bound also falls as one over k squared.

Each update evaluates the loss gradient at v, applies the closed-form shrinkage, and checks the full gradient at the accepted weights. Dense gradients cost order n times d. Shrinkage and momentum require only vector operations. We use the same general-convex momentum schedule without restart. The rate concerns the composite objective gap, while our stopping rule checks the full gradient.

For this quadratic penalty, the shrink step has a closed form. The next slide shows exactly why our default FISTA and Nesterov iterates coincide.

中文：
FISTA 使用与 Xuyang 的实现相同的 t 递推和外推。复合目标误差上界同样按一除以 k 平方下降。

每次更新在 v 处计算损失梯度，执行闭式收缩，并在接受的权重处检查完整梯度。稠密梯度计算需要 n 乘 d 量级的工作。收缩和动量只需要向量运算。我们使用同样的一般凸动量安排，没有重启。收敛速度描述复合目标差距，停止规则则检查完整梯度。

对于这个二次正则项，收缩步骤有闭式解。下一页会精确说明为什么默认 FISTA 与 Nesterov 的迭代点一致。

### Slide 19: Nesterov-FISTA equivalence

[Speaker: WU Aoqi | FISTA and equivalence | 14:20-16:00 | 100s]

This equivalence follows directly from the update, so we can predict the overlapping curves before looking at the results.

[Follow the four equalities slowly.]

Start with the FISTA step using one over L f. Multiply numerator and denominator by L f. The numerator becomes L f times v minus the loss gradient, and the denominator becomes L f plus lambda.

Rewrite the numerator by adding and subtracting lambda v. This produces v minus a fraction. Its numerator is the loss gradient plus lambda v, and its denominator is L f plus lambda.

The bracketed gradient is the gradient of the full objective. The denominator is L. We have therefore recovered the Nesterov gradient step exactly.

To obtain identical sequences, we also need identical initial weights and the same momentum recurrence. Both conditions hold in our code. The equality then carries from one update to the next, apart from floating-point rounding.

The conclusion depends on this isotropic L2 penalty and these step sizes. It does not establish an equivalence for arbitrary penalties or parameter choices.

ADMM also separates the loss and penalty, but introduces an equality constraint and a dual variable. ZHAO Jixin will explain that method next.

[Hand over to ZHAO Jixin at 16:00.]

中文：
这个等价关系可以直接从更新公式推出，因此在看到结果之前，我们就能预期曲线重合。

[缓慢沿着四行等式讲解。]

从使用一除以 L f 步长的 FISTA 更新开始，把分子分母同时乘以 L f。分子变成 L f 乘 v 减去损失梯度，分母变成 L f 加 lambda。

在分子中加上再减去 lambda v，就能改写成 v 减去一个分式。分式的分子是损失梯度加 lambda v，分母是 L f 加 lambda。

括号中的梯度正是完整目标的梯度，分母就是 L。因此我们精确恢复了 Nesterov 的梯度步。

要得到相同的整个序列，还需要相同的初始权重和动量递推。这两个条件在代码中都成立，所以除浮点舍入外，等式能从一次更新延续到下一次。

该结论依赖当前各向同性的 L2 正则项和步长，不能推广成任意正则项或参数下的等价关系。

ADMM 也分开处理损失与正则项，但会引入相等约束和对偶变量。下面请 ZHAO Jixin 解释这个方法。

[在 16:00 交给 ZHAO Jixin。]

### Slide 20: ADMM: two copies of the weights

[Speaker: ZHAO Jixin | ADMM | 16:00-17:00 | 60s]

I am ZHAO Jixin, and my solver is ADMM, the Alternating Direction Method of Multipliers. I will also compare the experimental results.

We give the loss and penalty separate copies of the weights. The loss uses w and the penalty uses z. The constraint w equals z preserves the original optimization problem.

We fix rho at one. Rho controls the equality penalty inside the algorithm, while lambda defines the model's regularization. They have different roles, and the Newton line-search step is another separate quantity.

[Point to w = z. At agreement, the split formulation is the original regularized logistic objective.]

中文：
我是 ZHAO Jixin，我的求解器是 ADMM，即交替方向乘子法。我也负责比较实验结果。

我们给损失和正则项各自设置一份权重。损失使用 w，正则项使用 z。约束 w 等于 z，保证它仍是原来的优化问题。

我们固定 rho 为一。rho 控制算法内部的相等约束惩罚，lambda 决定模型的正则化。两者作用不同，Newton 线搜索的步长又是另一项独立参数。

[指向 w 等于 z。两个副本一致时，分裂形式就是原来的正则化逻辑回归目标。]

### Slide 21: ADMM: the three updates

[Speaker: ZHAO Jixin | ADMM | 17:00-18:00 | 60s]

[Point to the constraint and then the three updates.]

The first update minimizes the logistic loss plus a quadratic term. That term pulls w toward z minus u. Logistic loss makes this subproblem nonlinear, so our implementation uses a warm-started Newton solve with backtracking.

The second update handles z. Both terms are quadratic in this variable, giving the direct shrinkage formula rho divided by rho plus lambda, multiplied by w plus u.

The third update adds the disagreement w minus z to u. This scaled dual variable carries constraint information into the next iteration. Repeatedly adjusting it encourages the copies to agree.

中文：
[先指向约束，再依次指向三条更新。]

第一步最小化逻辑损失与一个二次项之和。二次项将 w 拉向 z 减 u。逻辑损失使这个子问题成为非线性问题，因此实现采用带回溯的热启动 Newton 求解。

第二步处理 z。有关 z 的两项都是二次函数，所以有直接的收缩公式：rho 除以 rho 加 lambda，再乘 w 加 u。

第三步将不一致量 w 减 z 加到 u 上。这个缩放对偶变量把约束信息带入下一次迭代，持续调整它会推动两个副本一致。

### Slide 22: ADMM: work inside an outer update

[Speaker: ZHAO Jixin | ADMM | 18:00-19:00 | 60s]

For dense data, GD and the accelerated methods use order n times d work per gradient evaluation. Momentum adds vector operations, and the accelerated implementations also evaluate the full gradient for monitoring.

ADMM's Newton step forms a d-by-d Hessian and solves a linear system. The dense cost is order n times d squared plus d cubed per inner step. One outer update may need several such steps.

[Point to the inner-solve limits.]

Our implementation allows up to fifty Newton steps per outer update and uses an inner gradient tolerance of ten to the minus ten. A limit of fifty does not mean every update uses all fifty.

An ADMM outer update therefore contains a nonlinear inner solve followed by inexpensive vector operations. Even when the outer iteration count is modest, that inner work can dominate elapsed time.

中文：
对于稠密数据，GD 和加速方法每次梯度计算都需要 n 乘 d 量级的工作。动量增加向量运算，加速实现还会计算完整梯度来监控进展。

ADMM 的 Newton 步要构造 d 乘 d 的 Hessian，并求解线性系统。每个内部步的稠密成本是 n 乘 d 平方，加上 d 立方。一次外层更新可能需要多个这样的步骤。

[指向内部求解的限制。]

实现每次外层更新最多允许五十个 Newton 步，内部梯度容差为十的负十次方。上限为五十并不意味着每次都用满五十步。

因此，一次 ADMM 外层更新包含一次非线性内部求解，以及随后的低成本向量操作。即使外层次数不多，内部工作仍可能主导耗时。

### Slide 23: ADMM: residuals and stopping

[Speaker: ZHAO Jixin | ADMM | 19:00-20:00 | 60s]

To judge convergence, we must check agreement between the copies as well as stationarity of the original objective. This slide specifies those checks.

The primal residual r measures disagreement between w and z. The dual residual s tracks the change in z, scaled by rho. We require both residual checks, a successful inner solve, and the common full-gradient test.

All reported objectives evaluate the original F at the returned w. This keeps the comparison consistent across solvers.

I will now compare the four implementations on the same data, using both update counts and elapsed time.

[Continue as ZHAO Jixin into the experiments at 20:00.]

中文：
判断收敛时，既要检查两个副本的一致性，也要检查原始目标的平稳性。这一页具体说明这些检查。

原始残差 r 衡量 w 与 z 的不一致。对偶残差 s 衡量 z 的变化，并乘以 rho。我们要求两项残差检查、内部求解和共同的完整梯度检查全部通过。

所有报告的目标值都在返回的 w 上计算原始 F，保证不同求解器的比较一致。

下面我继续在相同数据上比较这四个实现，同时查看更新次数和耗时。

[在 20:00 由 ZHAO Jixin 继续讲实验。]

### Slide 24: Experimental setup

[Speaker: ZHAO Jixin | Experiments and comparison | 20:00-20:55 | 55s]

We use two datasets: synthetic classification data and the
breast-cancer dataset included with scikit-learn.

[point to the training and test columns]

We keep seventy-five percent for training and twenty-five percent for
testing. The split keeps similar class proportions.
The synthetic problem has fifty-one weights, including the bias.
The second has thirty-one.

We learn feature scaling from training data only.
All methods use the same split, zero start, penalty and stopping target.
The shared code checks the returned results and calculates the final
scores again.

Our timer covers the solver call, including setup and logging.
It excludes data loading and imports.
These choices tell us what the reported time includes.

中文：
我们使用两个数据集：合成分类数据，以及 scikit-learn 自带的乳腺癌数据集。

[指向训练集和测试集两列]

百分之七十五的数据用于训练，百分之二十五用于测试。划分时保持近似的类别比例。合成问题有五十一个权重，包含偏置；第二个问题有三十一个。

特征缩放参数只从训练数据中学习。所有方法使用相同的数据划分、全零起点、正则化和停止目标。共享代码检查返回的结果，并重新计算最终指标。

计时覆盖完整求解器调用，包括设置和日志记录，不包括数据加载和导入。这明确了所报告时间的统计范围。

### Slide 25: Synthetic: all four reach the target

[Speaker: ZHAO Jixin | Experiments and comparison | 20:55-22:10 | 75s]

All four methods pass on the synthetic data.
They reach the same objective, about zero point four nine three two.
Test accuracy is seventy-three point eight percent for every method.

[point to the updates column]

GD needs one hundred thirty-six updates.
Nesterov and FISTA each need one hundred thirty-eight.
ADMM needs one hundred ninety-five outer updates and passes its extra checks.

So GD reaches our target in slightly fewer updates here.
The word accelerated does not promise fewer steps on every dataset.
Our stopping rule checks the gradient, while the rate formula describes
the objective error.

[point to the wall-time column]

Times range from about seven to fifty-five milliseconds.
These are single recorded runs.
The small differences among the first three need repeated measurements
before we can call one reliably faster.

中文：
在合成数据上，四种方法全部通过检查。它们达到相同的目标值，约为 0.4932；测试准确率都为百分之七十三点八。

[指向更新次数一列]

GD 需要一百三十六次更新，Nesterov 和 FISTA 各需要一百三十八次。ADMM 需要一百九十五次外层更新，并通过额外检查。

所以，这里 GD 用略少的更新就达到了目标。“加速”并不保证在每个数据集上都用更少的步骤。我们的停止规则检查梯度，而收敛速度公式描述目标函数误差。

[指向总耗时一列]

时间约为七到五十五毫秒。这些都是单次运行的记录。前三种方法之间的小差异需要重复测量后，才能判断谁稳定地更快。

### Slide 26: Synthetic objective versus iteration

[Speaker: ZHAO Jixin | Experiments and comparison | 22:10-23:00 | 50s]

The horizontal axis counts updates. The vertical axis shows the original
objective, including the penalty.
Lower values mean a better fit to this training objective.

[point to the axes, then the start of the curves]

Every method starts with zero weights.
At that point, the objective is log two.
The curves drop early and then approach the same level.

Nesterov and FISTA overlap. Aoqi showed that their update formulas match
with our settings, so this is what we expect.

[point to the overlapping curves and the tail]

The vertical axis uses an ordinary linear scale.
Tiny differences near the end are hard to see.
We use the gradient values in the table to decide whether a run has
reached the target.

中文：
横轴是更新次数，纵轴是包含正则项的原始目标函数。数值越低，表示这个训练目标拟合得越好。

[指向坐标轴，再指向曲线起点]

每种方法都从全零权重开始，此时目标值是 log 2。曲线在前期下降，之后趋于相同的水平。

Nesterov 与 FISTA 重合。Aoqi 已经说明，在当前设置下它们的更新公式相同，因此这是预期中的结果。

[指向重合曲线和末端]

纵轴使用普通的线性刻度，末端的微小差异很难看出来。我们用表格中的梯度值判断是否达到了目标。

### Slide 27: Synthetic objective versus time

[Speaker: ZHAO Jixin | Experiments and comparison | 23:00-23:55 | 55s]

This plot keeps the same objective values, but changes the horizontal
axis from updates to time.

[point to the time axis and the ADMM curve]

ADMM takes longer in this run because it performs inner Newton solves.
The other three finish close together.
Counting outer updates alone would miss this extra work.

The curve uses timestamps recorded inside the solver.
The table times the complete call from outside.
That includes work at the end, so the two time measurements can differ
slightly.

Checks and logging also take time.
On a small dataset, they can affect the result.
We would repeat these measurements with fixed thread settings before
ranking small differences. For now, this plot describes the saved run.

中文：
这张图保留相同的目标函数值，把横轴从更新次数改成时间。

[指向时间轴和 ADMM 曲线]

ADMM 在本次运行中更慢，因为它执行内部 Newton 求解。其他三种方法的完成时间比较接近。只数外层更新次数，会漏掉这些额外工作。

曲线使用求解器内部记录的时间戳，表格则从外部计时整个调用，包括结束时的工作，因此两种时间测量可能略有不同。

检查和日志记录也会耗时。在小数据集上，它们可能影响结果。我们需要固定线程设置并重复测量，才能对细小差异排序。目前这张图描述的是已保存的这次运行。

### Slide 28: Breast cancer: the budget ends first

[Speaker: ZHAO Jixin | Experiments and comparison | 23:55-25:10 | 75s]

The second dataset gives a different result.
Every method uses all four hundred updates.
None reaches our gradient target of ten to the minus six.

[point to the updates and gradient columns]

GD has a gradient size of about one point seven times ten to the minus
three. Nesterov and FISTA are around four point two times ten to the
minus five. ADMM is close to those two, but also above the target.

All four still have the same test accuracy: ninety-six point five percent.
Their class predictions can agree even while training is unfinished.

Nesterov and FISTA reach a lower objective than GD within this budget.
That is a useful measured difference, even though none passes the
requested stopping check.

The report also has a longer ADMM check.
It uses a different update budget, so we keep it outside this table.
Here, we compare all four under the same limit.

中文：
第二个数据集给出了不同的结果。每种方法都用完四百次更新，没有一种达到十的负六次方这个梯度目标。

[指向更新次数和梯度两列]

GD 的梯度范数约为 1.7 乘十的负三次方。Nesterov 和 FISTA 约为 4.2 乘十的负五次方。ADMM 与后两者接近，但也高于目标。

四种方法的测试准确率仍然相同，都是百分之九十六点五。训练尚未完成时，它们的类别预测就可能已经一致。

在当前预算内，Nesterov 和 FISTA 的目标值低于 GD。这是一个有意义的实测差异，即使它们都未通过要求的停止检查。

报告里还有一次更长的 ADMM 检查。它采用不同的更新预算，所以没有放进这张表。这里四种方法都在相同上限下比较。

### Slide 29: Breast-cancer objective versus iteration

[Speaker: ZHAO Jixin | Experiments and comparison | 25:10-25:55 | 45s]

The objective drops quickly and then looks almost flat.
Looking at the graph alone, we might think the methods have finished.

[point to the flat tail]

The gradient sizes in the previous table are still above our target.
The scale hides the small changes that remain.

The graph helps us see the broad pattern.
The table answers the more precise question: did the stopping check pass?
We need both views to describe these runs.

A future plot could show the gap from a carefully checked reference
value on a logarithmic scale.
That would make small differences near the end easier to see.

中文：
目标函数先快速下降，随后看起来几乎变平。只看图时，我们可能以为这些方法已经完成训练。

[指向平坦的曲线末端]

上一张表里的梯度范数仍然高于目标。当前刻度隐藏了剩余的小变化。

图帮助我们看整体趋势；表格回答更精确的问题：是否通过停止检查？描述这些运行需要同时看两者。

以后可以用经过仔细验证的参考值，画出目标值之差，并采用对数刻度。这样更容易看清末端的微小差异。

### Slide 30: Comparison and solver choices

[Speaker: ZHAO Jixin | Experiments and comparison | 25:55-27:30 | 95s]

[Point to the solver rows.]

All four solvers pass the gradient check on the synthetic data. Nesterov and FISTA overlap because their default updates match for this penalty.

On breast cancer, equal test accuracy appears before any method reaches the requested gradient target. Those runs describe progress within a fixed budget.

ADMM also shows why outer update counts alone miss important work. Its inner solves take time.

GD is a useful starting point for this problem.
Its update is easy to understand, and it solves the synthetic case well.

Nesterov uses recent movement to look ahead.
FISTA handles the penalty with a cheap shrink step.
With our settings, they follow the same path.
So these runs cannot show a difference in their progress toward the answer.

ADMM separates the loss and penalty.
This can help when separate parts are easier to handle.
Here, its inner Newton solve adds work.

[point to the practical conclusion]

We should compare time at the same training precision.
We should also check the work inside each update.
A method with fewer outer updates may still take longer.
The formulas help us understand the methods, and the measured results
show what happens in this setup.

中文：
[指向表格的四行。]

四种求解器都在合成数据上通过梯度检查。Nesterov 与 FISTA 重合，因为在当前正则项下，它们的默认更新相同。

在乳腺癌数据上，还没有方法达到要求的梯度目标时，测试准确率就已经相同。这些运行反映的是固定预算内的进展。

ADMM 也说明只看外层更新次数会漏掉重要工作，其内部求解会消耗时间。

对于这个问题，GD 是一个有用的起点。更新规则容易理解，在合成数据上也能很好地求解。

Nesterov 利用最近的移动方向向前看，FISTA 用便宜的收缩步骤处理正则项。在我们的设置下，它们沿同一条路径前进，因此这些运行无法显示两者在接近答案时的进展差异。

ADMM 把损失和正则项分开。在各部分单独处理更容易时，这可能有帮助。但在这里，内部 Newton 求解增加了计算工作。

[指向实际结论]

我们应该在相同训练精度下比较时间，也应该检查每次更新内部的工作。外层更新更少的方法，仍可能花更长时间。公式帮助我们理解方法，实测结果则说明当前设置下会发生什么。

### Slide 31: Limitations and next experiments

[Speaker: ZHAO Jixin | Experiments and comparison | 27:30-29:00 | 90s]

Our comparison uses two small datasets, one split per dataset, and one timing per method. The breast-cancer runs exhaust the budget before reaching the common target.

[Point to the limitations.]

The next comparison should give every method enough updates to reach the same precision. We could then measure time to that target.

For timing, we would fix the thread settings used by matrix operations, repeat each run, and report the variation. Additional random splits would test whether the pattern persists.

We could also vary step sizes and rho, or evaluate momentum restart. These are future experiments, not settings used in the results shown today.

The current evidence supports conclusions about these saved runs. It does not establish a universally fastest solver.

LIU Xuyang will close with our main finding and invite questions.

[Hand over to LIU Xuyang at 29:00.]

中文：
比较只使用两个小数据集，每个数据集一次划分，每种方法计时一次。乳腺癌数据的运行在达到共同目标前就耗尽了预算。

[指向局限性。]

下一轮比较应给每种方法足够的更新次数，使它们达到相同精度，再测量达到该目标所需的时间。

计时方面，我们会固定矩阵运算的线程设置，重复各次运行，并报告波动。额外的随机划分可以检验趋势是否稳定。

我们也可以改变步长和 rho，或评估动量重启。这些属于后续实验，不是今天结果采用的设置。

当前证据支持对这些已保存运行的判断，无法确定一个在所有情况下都最快的求解器。

下面请 LIU Xuyang 总结主要发现，并邀请提问。

[在 29:00 交给 LIU Xuyang。]

### Slide 32: Conclusion and questions

[Speaker: LIU Xuyang | Closing | 29:00-30:00 | 60s]

Thank you. Each of us presented one solver for the same model: Zeqing on GD, myself on Nesterov, Aoqi on FISTA, and Jixin on ADMM.

All four pass the gradient check on the synthetic problem, with different amounts of computational work. On breast cancer, matching test accuracy comes before the training target is met.

The comparison therefore needs a clear optimization certificate as well as prediction accuracy. Runtime must account for the work inside each update.

Thank you for listening. We welcome your questions. Each member can answer questions about their own solver, and Jixin can discuss the shared results.

[Finish at 30:00. Keep this slide visible for Q&A. Open hidden slides only when needed.]

中文：
谢谢。我们每人讲解了一种求解同一模型的方法：Zeqing 讲 GD，我讲 Nesterov，Aoqi 讲 FISTA，Jixin 讲 ADMM。

四种方法都在合成问题上通过梯度检查，但计算工作量不同。在乳腺癌数据上，测试准确率在达到训练目标之前就已经一致。

因此，比较既需要明确的优化精度保证，也需要预测准确率。运行时间还必须反映每次更新内部的工作。

谢谢大家，欢迎提问。每位组员可以回答自己负责的求解器问题，Jixin 可以解释共同实验结果。

[在 30:00 结束。问答期间保留此页，仅在需要时打开隐藏页。]

### Slide 33: Backup: ADMM residual thresholds

[Speaker: ZHAO Jixin | Q&A backup | Q&A only | not timed]

[point to the relevant threshold]

The first check asks whether w and z agree closely enough.
The second checks how much z moves, scaled by rho.
We also require the full gradient check and a successful inner solve.
The final objective uses the returned w.

中文：
[指向问题所涉及的阈值]

第一项检查 w 和 z 是否足够接近。第二项检查 z 的变化量，并乘以 rho。我们还要求完整梯度通过检查，并且内部求解成功。最终目标值使用返回的 w 计算。

### Slide 34: Backup: references and code

[Speaker: LIU Xuyang | Q&A backup | Q&A only | not timed]

These references cover convex optimization, FISTA and ADMM.
Our local repository has the four methods and shared experiment code.

中文：
这些参考文献涵盖凸优化、FISTA 和 ADMM。本地代码仓库包含四种方法，以及共享的实验代码。
