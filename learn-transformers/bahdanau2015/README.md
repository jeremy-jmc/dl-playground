
# Neural Machine Translation by Jointly Learning to Align and Translate

English to French translation using Bahdanau et al. (2015) attention mechanism.

## Bahdanau et al. (2015)

### Model

<div align="center">
    <img src="https://lilianweng.github.io/posts/2018-06-24-attention/encoder-decoder-attention.png">
</div>

### Attention mechanism

$$
\begin{aligned}
\mathbf{x} &= [x_1, x_2, \dots, x_n] \\
\mathbf{y} &= [y_1, y_2, \dots, y_m]
\end{aligned}

\\

\boldsymbol{h}_i = [\overrightarrow{\boldsymbol{h}}_i^\top; \overleftarrow{\boldsymbol{h}}_i^\top]^\top, i=1,\dots,n

\\

\begin{aligned}
\mathbf{c}_t &= \sum_{i=1}^n \alpha_{t,i} \boldsymbol{h}_i & \small{\text{; Context vector for output }y_t}\\
\alpha_{t,i} &= \text{align}(y_t, x_i) & \small{\text{; How well two words }y_t\text{ and }x_i\text{ are aligned.}}\\
&= \frac{\exp(\text{score}(\boldsymbol{s}_{t-1}, \boldsymbol{h}_i))}{\sum_{i'=1}^n \exp(\text{score}(\boldsymbol{s}_{t-1}, \boldsymbol{h}_{i'}))} & \small{\text{; Softmax of some predefined alignment score.}}.
\end{aligned}

\\

\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \mathbf{v}_a^\top \tanh(\mathbf{W}_a[\boldsymbol{s}_t; \boldsymbol{h}_i])

\\

\boldsymbol{s}_t=f(\boldsymbol{s}_{t-1}, y_{t-1}, \mathbf{c}_t)
$$

- https://github.com/Maab-Nimir/Neural-Machine-Translation-by-Jointly-Learning-to-Align-and-Translate/tree/main
- https://github.com/mengjizhiyou/pytorch_model/blob/3c5e6eb5526b9f77bd3d43645aa954534e9385fb/bahdanau_attention.py
- https://github.com/sooftware/attentions/tree/master
- https://machinelearningmastery.com/the-bahdanau-attention-mechanism/