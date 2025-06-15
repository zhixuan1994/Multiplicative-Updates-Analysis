# Paper:
Multiplicative Updates Analysis for Quadratic Programming with Convex Domain Constraint

Many problems in statistical learning and neural computation involve linear optimizations with convex domain constraints. In this article, we study a large set of optimization problems in quadratic programming where the optimization is confined to any closed convex domain in $\mathbb R^p$, with $p\ge1$. We extend Sha et al.'s multiplicative updates for quadratic programming with nonnegative constraint to arbitrary closed convex domain constraint. Compared to existing methods for the same problems, our novel numerical scheme improves the value of the objective function iteratively and converges monotonically to the global optimizer, and we achieve closed form updates without the need of tuning. As an advantage to other multiplicative updates methods used in the machine learning literature, our algorithm can be easily hard-coded in any  languages, such as Python, $R$ and $MATLAB$. We show that this algorithm provides solutions to linear regularizations with any convex penalty functions. Examples include ridge, lasso, elastic net and $L^p$ ($p\ge1$) penalties. We also provide convergence analysis, simulation study with coding in Python to show the consistency and simplicity of our algorithms.  

Zhixuan Jia, 00034737@whu.edu.cn; Catherine Ma, zmbg2022@mymail.pomona.edu; Qidi Peng, qidi.peng@cgu.edu.

# Corresponding author:
Qidi Peng
