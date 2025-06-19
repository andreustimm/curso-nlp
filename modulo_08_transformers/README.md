[⬅️ Voltar para o índice do curso](../README.md)

# Módulo 8: Transformers e Modelos Pré-treinados

## 🎯 Objetivos do Módulo

Ao final deste módulo, você será capaz de:
- Compreender a arquitetura Transformer em detalhes
- Utilizar modelos pré-treinados (BERT, GPT, RoBERTa)
- Implementar fine-tuning para tarefas específicas
- Aplicar transfer learning em NLP
- Usar a biblioteca Transformers do Hugging Face
- Otimizar modelos grandes para produção

## 📚 Conteúdo Teórico

### 1. Arquitetura Transformer

#### 1.1 "Attention Is All You Need"
Paper revolucionário (2017) que introduziu arquitetura baseada apenas em attention.

**Motivação:**
- RNNs são sequenciais (não paralelizam)
- CNNs têm receptive field limitado
- Attention pode capturar dependências longas

#### 1.2 Multi-Head Attention

**Self-Attention:**
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

**Multi-Head:**
```
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Vantagens:**
- Diferentes tipos de relações
- Paralelização completa
- Interpretabilidade via attention weights

#### 1.3 Positional Encoding
Transformers não têm noção inerente de posição.

**Sinusoidal Encoding:**
```
PE(pos,2i) = sin(pos/10000^(2i/d_model))
PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
```

#### 1.4 Arquitetura Completa
```
Input → Embedding + Positional Encoding
↓
Multi-Head Attention
↓
Add & Norm
↓
Feed Forward Network
↓
Add & Norm
↓
(Repetir N vezes)
↓
Output
```

### 2. BERT (Bidirectional Encoder Representations from Transformers)

#### 2.1 Características Principais
- **Bidirectional**: Vê contexto completo
- **Pre-training**: MLM + NSP
- **Encoder-only**: Baseado em encoder do Transformer
- **Fine-tuning**: Adaptável para tarefas downstream

#### 2.2 Pre-training Tasks

**Masked Language Model (MLM):**
- 15% dos tokens são mascarados
- 80% → [MASK], 10% → palavra aleatória, 10% → original
- Modelo prediz tokens mascarados

**Next Sentence Prediction (NSP):**
- Prediz se frase B segue frase A
- 50% casos positivos, 50% negativos
- [CLS] token para classificação

#### 2.3 Tokens Especiais
- **[CLS]**: Início da sequência (classificação)
- **[SEP]**: Separador entre sentenças
- **[MASK]**: Token mascarado
- **[PAD]**: Padding
- **[UNK]**: Palavra desconhecida

#### 2.4 Variações do BERT
- **RoBERTa**: Remove NSP, dynamic masking
- **ALBERT**: Parameter sharing, factorized embeddings
- **DistilBERT**: Destilação, 60% menor
- **ELECTRA**: Discriminative pre-training

### 3. GPT (Generative Pre-trained Transformer)

#### 3.1 Arquitetura
- **Decoder-only**: Baseado em decoder do Transformer
- **Autoregressive**: Prediz próxima palavra
- **Unidirectional**: Vê apenas contexto anterior

#### 3.2 Evolução da Família GPT

**GPT-1 (2018):**
- 117M parâmetros
- Unsupervised pre-training + supervised fine-tuning

**GPT-2 (2019):**
- 1.5B parâmetros
- Zero-shot task transfer
- Controvérsia por não liberação inicial

**GPT-3 (2020):**
- 175B parâmetros
- Few-shot learning
- Emergent abilities

**GPT-4 (2023):**
- Multimodal
- Melhor raciocínio
- Mais seguro

#### 3.3 In-Context Learning
Capacidade de aprender tarefas apenas com exemplos no prompt.

**Zero-shot**: Apenas descrição da tarefa
**One-shot**: Um exemplo
**Few-shot**: Poucos exemplos

### 4. Transfer Learning em NLP

#### 4.1 Paradigma
1. **Pre-training**: Modelo geral em dados não rotulados
2. **Fine-tuning**: Adaptar para tarefa específica
3. **Inference**: Usar modelo adaptado

#### 4.2 Estratégias de Fine-tuning

**Feature-based:**
- Congela pesos do modelo pré-treinado
- Usa representações como features
- Adiciona classifier simples

**Fine-tuning:**
- Ajusta todos os pesos
- Learning rate menor que pre-training
- Regularização forte

**Gradual unfreezing:**
- Descongela camadas gradualmente
- Começar pelas últimas camadas
- Evita catastrophic forgetting

#### 4.3 Task-specific Heads
- **Classification**: [CLS] → Dense → Softmax
- **Token classification**: Each token → Dense → Softmax
- **Question Answering**: Start/End positions
- **Sequence generation**: Language modeling head

### 5. Hugging Face Transformers

#### 5.1 Biblioteca Principal
Padroniza uso de modelos pré-treinados.

**Componentes:**
- **Models**: Implementações de arquiteturas
- **Tokenizers**: Pré-processamento consistente
- **Pipelines**: Interface high-level
- **Datasets**: Carregamento padronizado

#### 5.2 Pipeline Básico
```python
from transformers import pipeline

# Análise de sentimentos
classifier = pipeline("sentiment-analysis")
result = classifier("I love this course!")

# Question answering
qa = pipeline("question-answering")
result = qa(question="What is NLP?", context="NLP is...")
```

#### 5.3 Fine-tuning com Trainer
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### 6. Otimização para Produção

#### 6.1 Model Compression

**Pruning:**
- Remove pesos menos importantes
- Structured vs unstructured
- Magnitude-based vs gradient-based

**Quantization:**
- Reduz precisão (FP32 → INT8)
- Post-training vs quantization-aware training
- Pode manter >95% da performance

**Distillation:**
- Treina modelo menor (student) para imitar maior (teacher)
- Soft targets preserve more information
- DistilBERT, TinyBERT, etc.

#### 6.2 Inference Optimization

**ONNX:**
- Open Neural Network Exchange
- Framework-agnostic
- Hardware-specific optimizations

**TensorRT:**
- NVIDIA optimization library
- Kernel fusion
- Mixed precision

**Dynamic Batching:**
- Agrupa requisições
- Amortiza overhead
- Aumenta throughput

## 🛠 Implementações Práticas

### Fine-tuning BERT
```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer

# Carregar modelo e tokenizer
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=2
)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenização
def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        truncation=True, 
        padding=True,
        max_length=512
    )

# Fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
```

### GPT para Geração
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Geração de texto
input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=3,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
```

### Custom Model com Transformers
```python
from transformers import BertModel
import torch.nn as nn

class CustomBERT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
```

## 📊 Exercícios Práticos

### Exercício 1: Fine-tuning BERT para Classificação
- Dataset de análise de sentimentos
- Comparar com modelos do módulo anterior
- Análise de attention weights

### Exercício 2: GPT para Geração de Texto
- Fine-tuning em corpus específico
- Diferentes estratégias de decodificação
- Avaliação de qualidade (BLEU, perplexity)

### Exercício 3: Question Answering com BERT
- Dataset SQuAD em português
- Implementar pipeline completo
- Avaliar em diferentes domínios

### Exercício 4: Model Compression
- Aplicar distillation em BERT
- Comparar accuracy vs speed
- Deploy otimizado

## 🎯 Próximos Passos

No **Módulo 9**, aplicaremos transformers em tarefas avançadas como NER, sumarização e tradução.

---

**Dica**: Execute o notebook `08_transformers_modelos_pretreinados.ipynb` para experimentar com modelos state-of-the-art! 

[⬅️ Voltar para o índice do curso](../README.md) 