# G8 math gate - answer key

## 1. Two-layer MLP gradient

For input dimension \(d\), hidden width \(h\), and class count \(c\):

- \(W_1\in\mathbb R^{h\times d}\)
- \(W_2\in\mathbb R^{c\times h}\)

Let \(e_y\) be the one-hot target and

\[
\delta_2=p-e_y.
\]

Then

\[
\frac{\partial L}{\partial W_2}=\delta_2h^\top
\]

and

\[
\delta_1=(W_2^\top\delta_2)\odot\mathbf 1[z_1>0],
\qquad
\frac{\partial L}{\partial W_1}=\delta_1x^\top.
\]

## 2. Rank and nullspace

The third row is the sum of the first two, while the first two rows are
independent. Therefore:

- \(\operatorname{rank}(A)=2\)
- \(\dim\operatorname{null}(A)=3-2=1\)
- one null vector is \((-1,-1,1)^\top\)
- \(A\) is not invertible because it is square but not full rank

## 3. Expectation, variance, and MAP

### 3a

\[
E[X]=0(1/4)+2(3/4)=3/2.
\]

\[
E[X^2]=0^2(1/4)+2^2(3/4)=3.
\]

\[
\operatorname{Var}(X)=3-(3/2)^2=3/4.
\]

### 3b

Likelihood precision is \(2/1=2\). Prior precision is \(1/4\).
The posterior precision is \(2+1/4=9/4\).

\[
\mu_{\text{MAP}}
=
\frac{(2+4)/1 + 0/4}{2+1/4}
=
\frac{6}{9/4}
=
\frac{8}{3}.
\]
