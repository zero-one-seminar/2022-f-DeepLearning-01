---

marp: true
math: katex
header: "01ゼミ 深層学習 活性化関数"
footer: "by リュカ"
theme: 01semi
paginate: true

---

<!-- p39-52 活性化関数 -->
<!-- class: title  -->
# 3章 ニューラルネットワーク
高柳海斗(リュカ)

---
<!-- class: slides  -->
# ニューラルネットワークとは

<div style = 'float:left;width:50%;'>

- パーセプトロンの応用
- 重みの自動決定ができる

中間層は隠れ層ということも多い

</div>

<div style = 'float:right;width:50%;text-align:center;'>

  ![h:390px](2layer.svg)
  2層ネットワーク

</div>

---
# パーセプトロン
<div style = 'float:left;width:50%;'>

数式で表すと

$y = \begin{cases}
  0 & (b+w_1x_1+w_2x_2 \leq 0)\\
  1 & (b+w_1x_1+w_2x_2 > 0)\\
\end{cases}$

</div>
<div style = 'float:right;width:50%;text-align:center;'>

  図で表すと
  ![h:390px](perceptron.svg)
</div>

---
# パーセプトロン
<div style = 'float:left;width:50%;'>

数式で表すと

$$
\begin{align*}
y &= \begin{cases}
  0 & (b+w_1x_1+w_2x_2 \leq 0)\\
  1 & (b+w_1x_1+w_2x_2 > 0)\\
\end{cases}\\
&=h(b+w_1x_1+w_2x_2)\\
\end{align*}
$$
$$
h(x) = \begin{cases}
  0 & (x \leq 0)\\
  1 & (x > 0)\\
\end{cases}
$$

</div>
<div style = 'float:right;width:50%;text-align:center;'>

  図で表すと
  ![h:390px](acti_func.svg)
</div>

---
# 活性化関数
入力信号の総和がどのように活性化するかを決める

パーセプトロンの場合はステップ関数
$$
h(x) = \begin{cases}
  0 & (x \leq 0)\\
  1 & (x > 0)\\
\end{cases}
$$

パーセプトロンの活性化関数を一般の非線形関数に拡張したものを
ニューラルネットワークと呼ぶ

---
# シグモイド関数
<div style = 'float:left;width:50%;'>

$$
h(x) = \frac{1}{1+\exp(-x)}
$$

シグモイド関数のいいところ
-  $0<h(x)<1$
-  $0.5$を中心に対称
-  滑らか・微分可能
</div>
<div style = 'float:right;width:50%;text-align:center;'>

  ![h:390px](sigmoid.png)
  シグモイド関数のグラフ
</div>

---

# ReLU

<div style = 'float:left;width:50%;'>

$$
h(x) = \begin{cases}
  x & (x > 0)\\
  0 & (x \leq 0)\\
\end{cases}
$$

ReLU関数のいいところ
-  計算コストが低い
-  シグモイドより性能が良くなることがある(2011 Xavier Glorot)
</div>
<div style = 'float:right;width:50%;text-align:center;'>

  ![h:390px](ReLU.png)
  ReLU関数のグラフ
</div>

---
# 実装してみよう
```python
import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
  return np.array(x>0, dtype = np.int)

fig, ax = plt.subplots()
x = np.linspace(-10, 10, 1000)
ax.plot(x, list(map(relu, x)))
ax.set_xlabel('$x$')
ax.set_ylabel('$h(x)$')
ax.grid()
fig.savefig("step.png")
```

---
# ブロードキャスト

## numpyの機能
```python
A = np.array([[1,2],[3,4]])
B = np.array([[10,20]])
A * B
#=> array([[ 10, 40],
#          [ 30, 80]])
```

---
# 実装してみよう
シグモイド
```python
def sigmoid(x):
  return 1 / (1+np.exp(-x))
```
ReLU
```python
def relu(x):
  return np.maximum(0,x)
```
---
# その他の活性化関数

<div style = 'float:left;width:50%;'>

## Leaky ReLU
$$
h(x) = \begin{cases}
  x & (x > 0)\\
  0.01 x & (x \leq 0)\\
\end{cases}
$$
## Swish
$$
h(x) = x\sigma (\beta x)\\
\sigma (\beta x) = \frac{1}{1+e^{-\beta x}}
$$

</div>
<div style = 'float:right;width:50%;'>

## ソフトサイン
$$
h(x) = \frac{x}{1+|x|}
$$

## Snake

$$
h(x) = x + \sin^2 x
$$
</div>

---

# 参考文献
- ゼロから作る Deep Learning
- Xavier Glorot; Antoine Bordes; Yoshua Bengio. “Deep Sparse Rectifier Neural Networks”. Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS-11) 15: 315-323.