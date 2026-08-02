# G8 math gate

Time limit: 45 minutes. Pen and paper. No notes after the 10-minute review.

Do not open `g8-math-gate-answer.md` until the timer expires.

## 1. Two-layer MLP gradient

Let

- \(z_1=W_1x+b_1\)
- \(h=\operatorname{ReLU}(z_1)\)
- \(z_2=W_2h+b_2\)
- \(p=\operatorname{softmax}(z_2)\)
- \(L=-\log p_y\)

State the shapes of \(W_1\) and \(W_2\), then derive:

1. \(\partial L/\partial W_2\)
2. \(\partial L/\partial W_1\)

Every outer product and elementwise product must be dimensionally clear.

## 2. Rank and nullspace

For

\[
A=
\begin{bmatrix}
1&2&3\\
0&1&1\\
1&3&4
\end{bmatrix},
\]

find:

1. \(\operatorname{rank}(A)\)
2. \(\dim\operatorname{null}(A)\)
3. one nonzero vector in \(\operatorname{null}(A)\)
4. whether \(A\) is invertible, with one-sentence justification

## 3. Expectation, variance, and MAP

### 3a. Expectation and variance

A random variable \(X\) has

- \(P(X=0)=1/4\)
- \(P(X=2)=3/4\)

Compute \(E[X]\) and \(\operatorname{Var}(X)\).

### 3b. Gaussian-prior MAP

Observations are \(x_1=2\) and \(x_2=4\), independently distributed as
\(\mathcal N(\mu,1)\). The prior is \(\mu\sim\mathcal N(0,4)\).

Compute the MAP estimate of \(\mu\). Show the likelihood precision, prior
precision, and resulting posterior mean.
