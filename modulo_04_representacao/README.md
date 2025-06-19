[⬅️ **Voltar**](../README.md)

# Módulo 4: Representação de Texto

## 🎯 Objetivos do Módulo

Ao final deste módulo, você será capaz de:
- Implementar Bag of Words (BoW) e suas variações
- Calcular e interpretar TF-IDF
- Treinar e utilizar word embeddings (Word2Vec, GloVe, FastText)
- Avaliar qualidade de representações vetoriais
- Aplicar redução de dimensionalidade em dados textuais
- Escolher a representação adequada para cada tarefa

## 📚 Conteúdo Teórico

### 1. Fundamentos da Representação Vetorial

#### 1.1 Por que Vetorizar Texto?
Algoritmos de machine learning trabalham com números, não com texto. A vetorização converte texto em representações numéricas preservando informação semântica.

**Desafios:**
- **Esparsidade**: Maioria dos valores são zero
- **Dimensionalidade**: Vocabulários grandes = vetores enormes
- **Semântica**: Capturar significado além de sintaxe
- **Eficiência**: Processamento rápido para grandes corpus

#### 1.2 Tipos de Representação
- **One-hot encoding**: Cada palavra = vetor binário
- **Contagem**: Frequência de palavras
- **TF-IDF**: Frequência ponderada por raridade
- **Embeddings**: Representações densas aprendidas

### 2. Bag of Words (BoW)

#### 2.1 Conceito Básico
Representa documento como "saco de palavras", ignorando ordem e estrutura.

**Características:**
- **Independência posicional**: Ordem não importa
- **Esparsidade**: Muitos zeros
- **Interpretabilidade**: Fácil de entender
- **Baseline**: Boa linha de base para comparação

#### 2.2 Variações do BoW

**Binary BoW:**
- 1 se palavra presente, 0 caso contrário
- Ignora frequência
- Útil para classificação de tópicos

**Count BoW:**
- Frequência absoluta de cada palavra
- Considera repetições
- Pode dar peso excessivo a palavras comuns

**Normalized BoW:**
- Frequências normalizadas por comprimento
- Evita viés por tamanho de documento
- Fórmulas: L1, L2, max normalization

#### 2.3 Implementação
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    max_features=5000,      # Vocabulário máximo
    ngram_range=(1, 2),     # Uni e bigramas
    stop_words='english',   # Remover stopwords
    min_df=2,               # Frequência mínima
    max_df=0.95             # Frequência máxima
)
```

### 3. TF-IDF (Term Frequency - Inverse Document Frequency)

#### 3.1 Motivação
BoW trata todas as palavras igualmente. TF-IDF pondera palavras por sua importância no corpus.

**Intuição:**
- Palavras frequentes no documento são importantes (TF)
- Palavras raras no corpus são distintivas (IDF)
- Combine ambas para medir relevância

#### 3.2 Fórmulas Matemáticas

**Term Frequency (TF):**
- TF(t,d) = (# de vezes que t aparece em d) / (# total de palavras em d)
- Log normalization: 1 + log(tf)
- Binary: 1 se presente, 0 caso contrário

**Inverse Document Frequency (IDF):**
- IDF(t) = log(N / df(t))
- N = número total de documentos
- df(t) = número de documentos contendo termo t

**TF-IDF Score:**
- TF-IDF(t,d) = TF(t,d) × IDF(t)
- Smoothed IDF: log(N / (1 + df(t))) + 1

#### 3.3 Interpretação
- **Valores altos**: Palavras frequentes no documento, raras no corpus
- **Valores baixos**: Palavras comuns ou ausentes
- **Zero**: Palavra não aparece no documento
- **Stopwords**: Geralmente têm IDF baixo

#### 3.4 Vantagens e Limitações

**Vantagens:**
- Reduz peso de palavras comuns
- Destaca termos distintivos
- Funciona bem para busca e classificação
- Interpretável e eficiente

**Limitações:**
- Ainda ignora ordem das palavras
- Não captura relações semânticas
- Esparsidade permanece
- Sensível ao tamanho do corpus

### 4. Word Embeddings

#### 4.1 Conceito
Representações densas de baixa dimensionalidade que capturam relações semânticas.

**Hipótese distribucional**: "Palavras que aparecem em contextos similares têm significados similares"

**Características:**
- **Densas**: Poucos zeros, muita informação
- **Baixa dimensionalidade**: 50-300 dimensões
- **Semântica**: Capturam relações de significado
- **Aprendizagem**: Extraídas de grandes corpus

#### 4.2 Word2Vec

**Arquiteturas:**
- **CBOW (Continuous Bag of Words)**: Prediz palavra central dado contexto
- **Skip-gram**: Prediz contexto dada palavra central

**Hiperparâmetros:**
- **vector_size**: Dimensionalidade (100-300)
- **window**: Tamanho da janela de contexto
- **min_count**: Frequência mínima das palavras
- **workers**: Paralelização
- **sg**: 0=CBOW, 1=Skip-gram

**Propriedades Matemáticas:**
- rei - homem + mulher ≈ rainha
- Paris - França + Itália ≈ Roma
- walking - walk + swim ≈ swimming

#### 4.3 GloVe (Global Vectors)
Combina estatísticas globais (matriz de co-ocorrência) com aprendizado local.

**Vantagens sobre Word2Vec:**
- Usa estatísticas globais do corpus
- Treinamento mais estável
- Melhor performance em tarefas de analogia

#### 4.4 FastText
Extensão do Word2Vec que considera subpalavras (character n-grams).

**Vantagens:**
- Lida com palavras fora do vocabulário (OOV)
- Útil para idiomas com morfologia rica
- Compartilha informação entre palavras similares

### 5. Redução de Dimensionalidade

#### 5.1 Por que Reduzir Dimensões?
- **Visualização**: Plotar em 2D/3D
- **Eficiência**: Menos parâmetros para treinar
- **Ruído**: Remover dimensões irrelevantes
- **Storage**: Economizar memória

#### 5.2 Técnicas Lineares

**PCA (Principal Component Analysis):**
- Encontra direções de maior variância
- Componentes são combinações lineares de features
- Preserva variância global

**SVD (Singular Value Decomposition):**
- Factorização de matriz
- Base para Latent Semantic Analysis (LSA)
- Identifica tópicos latentes

#### 5.3 Técnicas Não-lineares

**t-SNE (t-Distributed Stochastic Neighbor Embedding):**
- Preserva estrutura local
- Excelente para visualização
- Não preserva distâncias globais

**UMAP (Uniform Manifold Approximation):**
- Mais rápido que t-SNE
- Preserva melhor estrutura global
- Bom para visualização e pré-processamento

### 6. Avaliação de Representações

#### 6.1 Métricas Intrínsecas

**Analogias:**
- a : b :: c : ?
- Teste: rei : homem :: rainha : mulher
- Datasets: Google analogies, BATS

**Similaridade:**
- Correlação com julgamentos humanos
- Datasets: SimLex-999, WordSim-353
- Métricas: Spearman correlation

**Outlier Detection:**
- Identificar palavra que não pertence ao grupo
- Exemplo: [gato, cachorro, carro, gato]

#### 6.2 Métricas Extrínsecas
- **Classificação de texto**: Accuracy, F1-score
- **Análise de sentimentos**: Correlação com labels
- **NER**: Precision, Recall, F1
- **Parsing**: Attachment accuracy

#### 6.3 Análise Qualitativa
- **Vizinhos mais próximos**: Palavras similares
- **Visualização**: t-SNE, UMAP plots
- **Projeções**: Operações matemáticas
- **Bias detection**: Preconceitos nos embeddings

## 🛠 Implementações Práticas

### Bag of Words com Scikit-learn
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# BoW simples
bow_vectorizer = CountVectorizer(max_features=1000)
bow_matrix = bow_vectorizer.fit_transform(corpus)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
```

### Word2Vec com Gensim
```python
from gensim.models import Word2Vec

# Treinar modelo
model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1  # Skip-gram
)

# Usar modelo
vector = model.wv['palavra']
similares = model.wv.most_similar('palavra', topn=10)
```

### FastText
```python
from gensim.models import FastText

model = FastText(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1
)

# Vantagem: lida com OOV
vector_oov = model.wv['palavrainexistente']
```

## 📊 Exercícios Práticos

### Exercício 1: Comparação BoW vs TF-IDF
- Implementar ambas representações
- Comparar esparsidade e performance
- Analisar palavras com maior peso

### Exercício 2: Treinamento de Word2Vec
- Treinar embeddings em corpus português
- Explorar analogias e similaridades
- Visualizar embeddings com t-SNE

### Exercício 3: Avaliação de Embeddings
- Implementar métricas de avaliação
- Comparar Word2Vec, GloVe e FastText
- Analisar bias nos embeddings

### Exercício 4: Redução de Dimensionalidade
- Aplicar PCA em matriz TF-IDF
- Visualizar documentos em espaço reduzido
- Comparar PCA vs t-SNE vs UMAP

### Exercício 5: Sistema de Busca
- Implementar busca semântica
- Comparar diferentes representações
- Avaliar relevância dos resultados

## 🚨 Armadilhas Comuns

### Problemas com BoW/TF-IDF
- **Esparsidade extrema**: Vocabulário muito grande
- **OOV words**: Palavras não vistas no treino
- **Perda de ordem**: Ignore estrutura sintática
- **Polissemia**: Uma palavra, múltiplos significados

### Problemas com Embeddings
- **Dados insuficientes**: Precisam de corpus grandes
- **Bias**: Refletem preconceitos dos dados
- **Ambiguidade**: Uma representação por palavra
- **Avaliação**: Métricas nem sempre refletem qualidade

## 📖 Leituras Complementares

### Papers Fundamentais
- "Efficient Estimation of Word Representations in Vector Space" (Word2Vec)
- "GloVe: Global Vectors for Word Representation"
- "Enriching Word Vectors with Subword Information" (FastText)
- "Distributed Representations of Words and Phrases and their Compositionality"

### Recursos Online
- [Word2Vec Tutorial](https://rare-technologies.com/word2vec-tutorial/)
- [GloVe Project Page](https://nlp.stanford.edu/projects/glove/)
- [FastText Documentation](https://fasttext.cc/)

## 🎯 Próximos Passos

No **Módulo 5**, vamos aplicar essas representações para classificação de texto usando algoritmos de machine learning.

---

**Dica**: Execute o notebook `04_representacao_texto.ipynb` para experimentar todas as técnicas! 

[⬅️ **Voltar**](../README.md) 