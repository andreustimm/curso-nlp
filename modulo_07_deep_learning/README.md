[⬅️ **Voltar**](../README.md)

# Módulo 7: Deep Learning para NLP

## 🎯 Objetivos do Módulo

Ao final deste módulo, você será capaz de:
- Implementar redes neurais feedforward para texto
- Construir e treinar RNNs, LSTMs e GRUs
- Aplicar CNNs para classificação de texto
- Compreender e implementar attention mechanisms
- Otimizar arquiteturas neurais para NLP
- Debugar e melhorar modelos de deep learning

## 📚 Conteúdo Teórico

### 1. Redes Neurais Feedforward

#### 1.1 Conceitos Básicos
- **Perceptron**: Unidade básica
- **Multi-layer Perceptron**: Múltiplas camadas
- **Backpropagation**: Treinamento via gradiente
- **Universal approximation**: Capacidade de aproximar qualquer função

#### 1.2 Aplicação em Texto
```python
# Arquitetura simples
Input (TF-IDF) → Hidden Layer → ReLU → Output Layer → Softmax
```

**Vantagens:**
- Simples de implementar
- Baseline forte
- Rápido treinamento

**Limitações:**
- Ignora ordem das palavras
- Features fixas
- Sem memória de longo prazo

### 2. Redes Neurais Recorrentes (RNN)

#### 2.1 Motivação
Texto é sequencial. RNNs processam sequências mantendo estado interno.

#### 2.2 Arquitetura RNN
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

**Componentes:**
- **Estado oculto**: h_t (memória)
- **Pesos recorrentes**: W_hh
- **Pesos de entrada**: W_xh

#### 2.3 Problemas das RNNs
- **Vanishing gradients**: Gradientes diminuem exponencialmente
- **Exploding gradients**: Gradientes explodem
- **Dependências longas**: Dificuldade em capturar contexto distante

### 3. Long Short-Term Memory (LSTM)

#### 3.1 Arquitetura LSTM
Resolve problemas de RNN através de gates:

**Forget Gate:**
f_t = σ(W_f * [h_{t-1}, x_t] + b_f)

**Input Gate:**
i_t = σ(W_i * [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)

**Cell State Update:**
C_t = f_t * C_{t-1} + i_t * C̃_t

**Output Gate:**
o_t = σ(W_o * [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)

#### 3.2 Vantagens
- Captura dependências longas
- Controla fluxo de informação
- Gradientes mais estáveis
- Versátil para muitas tarefas

### 4. Gated Recurrent Unit (GRU)

#### 4.1 Simplificação da LSTM
- Menos parâmetros que LSTM
- Combina forget e input gates
- Performance similar à LSTM
- Treinamento mais rápido

#### 4.2 Fórmulas GRU
```
r_t = σ(W_r * [h_{t-1}, x_t])    # Reset gate
z_t = σ(W_z * [h_{t-1}, x_t])    # Update gate
h̃_t = tanh(W * [r_t * h_{t-1}, x_t])
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
```

### 5. Redes Neurais Convolucionais (CNN)

#### 5.1 CNNs para Texto
Aplicam filtros convolucionais sobre embeddings de palavras.

**Arquitetura:**
```
Input Embeddings → Conv1D → ReLU → MaxPool → Dense → Softmax
```

#### 5.2 Vantagens
- Detecta n-gramas automaticamente
- Translation invariant
- Paralelização eficiente
- Bom para classificação

#### 5.3 Hiperparâmetros
- **Filter sizes**: [2,3,4,5] para capturar diferentes n-gramas
- **Number of filters**: 100-200 por tamanho
- **Pooling**: Max pooling mais comum

### 6. Attention Mechanisms

#### 6.1 Motivação
RNNs têm bottleneck no último estado. Attention permite acesso a todos estados.

#### 6.2 Attention Básico
```
e_ij = a(s_{i-1}, h_j)           # Energy/score
α_ij = softmax(e_ij)             # Attention weights
c_i = Σ α_ij * h_j               # Context vector
```

#### 6.3 Tipos de Attention
- **Additive (Bahdanau)**: MLP para computar scores
- **Multiplicative (Luong)**: Produto escalar
- **Self-attention**: Query, key, value da mesma sequência

#### 6.4 Multi-Head Attention
- Múltiplas "cabeças" de attention
- Capturam diferentes tipos de relações
- Base dos Transformers

### 7. Arquiteturas Avançadas

#### 7.1 Bidirectional RNN/LSTM
- Processa sequência nas duas direções
- Combina informação passada e futura
- Melhor para tarefas onde contexto completo está disponível

#### 7.2 Sequence-to-Sequence (Seq2Seq)
- Encoder-Decoder architecture
- Encoder: RNN que processa entrada
- Decoder: RNN que gera saída
- Aplicações: tradução, sumarização

#### 7.3 Attention-based Seq2Seq
- Adiciona attention ao Seq2Seq
- Resolver bottleneck do encoder
- Permite alinhamento entrada-saída

## 🛠 Implementações Práticas

### LSTM com TensorFlow/Keras
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### CNN para Texto
```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

### Attention Layer
```python
import tensorflow as tf

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, hidden_states):
        # hidden_states: (batch_size, time_steps, features)
        score = self.V(tf.nn.tanh(self.W(hidden_states)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
```

## 📊 Exercícios Práticos

### Exercício 1: Comparação de Arquiteturas
- Implementar MLP, RNN, LSTM, GRU, CNN
- Avaliar em dataset de classificação
- Analisar tempo de treinamento vs performance

### Exercício 2: Análise de Sentimentos com LSTM
- Dataset de reviews
- Bidirectional LSTM
- Attention visualization

### Exercício 3: CNN para Classificação de Texto
- Múltiplos filter sizes
- Regularização (dropout, batch norm)
- Comparar com RNNs

### Exercício 4: Seq2Seq com Attention
- Tarefa de sumarização
- Implementar attention mechanism
- Visualizar attention weights

## 🎯 Próximos Passos

No **Módulo 8**, exploraremos Transformers e modelos pré-treinados como BERT e GPT.

---

**Dica**: Execute o notebook `07_deep_learning_nlp.ipynb` para implementar todas as arquiteturas! 

[⬅️ **Voltar**](../README.md) 