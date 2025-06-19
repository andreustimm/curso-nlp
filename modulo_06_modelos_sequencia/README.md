[⬅️ Voltar](../README.md)

# Módulo 6: Modelos de Sequência

## 🎯 Objetivos do Módulo

Ao final deste módulo, você será capaz de:
- Implementar modelos de n-gramas para linguagem
- Compreender e aplicar Hidden Markov Models (HMM)
- Utilizar Conditional Random Fields (CRF)
- Aplicar modelos sequenciais em POS tagging
- Implementar Named Entity Recognition (NER)
- Avaliar modelos de sequência adequadamente

## 📚 Conteúdo Teórico

### 1. Modelos de N-gramas

#### 1.1 Conceito Básico
Modelos de linguagem que predizem próxima palavra baseado em n-1 palavras anteriores.

**Tipos:**
- **Unigrama**: P(wi) - independência total
- **Bigrama**: P(wi|wi-1) - depende da palavra anterior
- **Trigrama**: P(wi|wi-1,wi-2) - depende das duas anteriores

#### 1.2 Estimação de Probabilidades
P(wi|wi-n+1...wi-1) = count(wi-n+1...wi) / count(wi-n+1...wi-1)

**Problemas:**
- Sparse data: combinações não vistas
- Zero probabilities: palavras OOV
- Data sparsity aumenta com n

#### 1.3 Suavização (Smoothing)
- **Add-one (Laplace)**: Adiciona 1 a todas contagens
- **Add-k**: Adiciona k < 1
- **Good-Turing**: Baseado em frequências de frequências
- **Kneser-Ney**: Estado da arte para n-gramas

### 2. Hidden Markov Models (HMM)

#### 2.1 Componentes
- **Estados ocultos**: O que queremos inferir
- **Observações**: O que vemos (palavras)
- **Probabilidades de transição**: P(si|si-1)
- **Probabilidades de emissão**: P(wi|si)

#### 2.2 Algoritmos Fundamentais

**Forward Algorithm:**
- Calcula probabilidade de sequência
- Dynamic programming
- O(T×N²) complexidade

**Viterbi Algorithm:**
- Encontra sequência mais provável de estados
- Decodificação MAP
- Usado em POS tagging

**Baum-Welch Algorithm:**
- Treinamento não supervisionado
- EM algorithm para HMMs
- Estima parâmetros dos dados

#### 2.3 Aplicações em NLP
- **POS Tagging**: Estados = tags gramaticais
- **Speech Recognition**: Estados = fonemas
- **Gene Prediction**: Estados = regiões genéticas

### 3. Conditional Random Fields (CRF)

#### 3.1 Motivação
- HMMs fazem assumption de independência forte
- CRFs modelam P(y|x) diretamente
- Permitem features arbitrárias
- Treinamento discriminativo

#### 3.2 Formulação Matemática
P(y|x) = (1/Z(x)) × exp(∑λifi(yi-1,yi,x,i))

**Componentes:**
- **Feature functions**: fi(yi-1,yi,x,i)
- **Weights**: λi
- **Partition function**: Z(x)

#### 3.3 Features em CRFs
- **Transition features**: Dependem de yi-1, yi
- **Observation features**: Dependem de yi, xi
- **Custom features**: Regex, gazeteers, etc.

### 4. Part-of-Speech (POS) Tagging

#### 4.1 Definição da Tarefa
Atribuir categoria gramatical a cada palavra.

**Tags comuns:**
- **NN**: Noun (substantivo)
- **VB**: Verb (verbo)
- **JJ**: Adjective (adjetivo)
- **RB**: Adverb (advérbio)
- **DT**: Determiner (determinante)

#### 4.2 Desafios
- **Ambiguidade**: "bank" = substantivo ou verbo
- **Unknown words**: Palavras não vistas
- **Context dependency**: "running water" vs "running fast"

#### 4.3 Abordagens
- **Rule-based**: Regras linguísticas
- **Stochastic**: HMM, CRF
- **Neural**: RNN, LSTM, Transformer

### 5. Named Entity Recognition (NER)

#### 5.1 Tipos de Entidades
- **PERSON**: Nomes de pessoas
- **LOCATION**: Lugares geográficos
- **ORGANIZATION**: Empresas, instituições
- **MISCELLANEOUS**: Outros (eventos, produtos)

#### 5.2 Esquemas de Anotação

**IOB (Inside-Outside-Begin):**
- B-PER: Início de pessoa
- I-PER: Continuação de pessoa
- O: Fora de entidade

**BILOU:**
- B: Begin, I: Inside, L: Last, O: Outside, U: Unit

#### 5.3 Features para NER
- **Lexical**: Palavra atual, prefixos, sufixos
- **Orthographic**: Maiúsculas, pontuação, números
- **Contextual**: Palavras vizinhas
- **Gazetteers**: Listas de nomes conhecidos

## 🛠 Implementações Práticas

### N-grama com NLTK
```python
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

# Preparar dados
train, vocab = padded_everygram_pipeline(3, corpus)

# Treinar modelo
lm = MLE(3)
lm.fit(train, vocab)

# Usar modelo
prob = lm.score("word", ["previous", "words"])
```

### HMM com NLTK
```python
from nltk.tag import hmm

# Treinar tagger
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(training_data)

# Usar tagger
tags = tagger.tag(["The", "cat", "sat"])
```

### CRF com sklearn-crfsuite
```python
import sklearn_crfsuite

# Definir features
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    return features

# Treinar CRF
crf = sklearn_crfsuite.CRF()
crf.fit(X_train, y_train)
```

## 📊 Exercícios Práticos

### Exercício 1: Modelo de Linguagem N-grama
- Implementar e comparar diferentes ordens
- Avaliar perplexidade
- Implementar suavização

### Exercício 2: POS Tagging com HMM
- Treinar tagger em corpus anotado
- Avaliar accuracy por tag
- Analisar erros comuns

### Exercício 3: NER com CRF
- Feature engineering para entidades
- Comparar diferentes esquemas de anotação
- Avaliar precision, recall, F1

### Exercício 4: Análise Comparativa
- HMM vs CRF para POS tagging
- Diferentes features para NER
- Error analysis detalhada

## 🎯 Próximos Passos

No **Módulo 7**, mergulharemos em deep learning para NLP com RNNs, LSTMs e attention.

---

**Dica**: Execute o notebook `06_modelos_sequencia.ipynb` para implementar todos os modelos!

[⬅️ Voltar](../README.md) 