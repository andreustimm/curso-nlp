# Módulo 3: Análise Estatística de Texto

## 🎯 Objetivos do Módulo

Ao final deste módulo, você será capaz de:
- Realizar análise de frequência de palavras e caracteres
- Implementar e interpretar n-gramas
- Calcular medidas de similaridade entre textos
- Aplicar análise de sentimentos básica
- Criar visualizações eficazes para dados textuais
- Detectar padrões e tendências em corpus textuais

## 📚 Conteúdo Teórico

### 1. Análise de Frequência

#### 1.1 Frequência de Palavras
A análise de frequência é fundamental para entender a distribuição de palavras em um corpus.

**Lei de Zipf**: A frequência de uma palavra é inversamente proporcional ao seu ranking.
- 1ª palavra mais frequente aparece ~2x mais que a 2ª
- 2ª palavra aparece ~2x mais que a 4ª
- Padrão: f(r) ∝ 1/r

**Métricas importantes:**
- **Frequência absoluta**: Contagem bruta de ocorrências
- **Frequência relativa**: Proporção em relação ao total
- **Vocabulário**: Número de palavras únicas
- **Hapax legomena**: Palavras que aparecem apenas uma vez

#### 1.2 Distribuições Estatísticas
- **Distribuição de Zipf**: Para frequência de palavras
- **Distribuição de Poisson**: Para eventos raros
- **Distribuição normal**: Para características contínuas

### 2. N-gramas e Colocações

#### 2.1 N-gramas
Sequências contíguas de n palavras que capturam contexto local.

**Tipos de n-gramas:**
- **Unigramas** (n=1): Palavras individuais
- **Bigramas** (n=2): Pares de palavras consecutivas
- **Trigramas** (n=3): Triplas de palavras
- **4-gramas** e superiores: Contextos mais longos

**Aplicações:**
- Detecção de frases idiomáticas
- Análise de estilo de escrita
- Modelos de linguagem
- Correção ortográfica

#### 2.2 Colocações
Combinações de palavras que aparecem juntas com frequência acima do esperado.

**Medidas de associação:**
- **PMI (Pointwise Mutual Information)**
- **Chi-quadrado (χ²)**
- **T-score**
- **Log-likelihood ratio**

### 3. Medidas de Similaridade

#### 3.1 Similaridade Baseada em Contagem
- **Jaccard**: |A ∩ B| / |A ∪ B|
- **Dice**: 2|A ∩ B| / (|A| + |B|)
- **Overlap**: |A ∩ B| / min(|A|, |B|)

#### 3.2 Similaridade Baseada em Vetores
- **Cosine similarity**: cos(θ) = A·B / (||A|| ||B||)
- **Distância euclidiana**: ||A - B||
- **Distância de Manhattan**: Σ|ai - bi|

#### 3.3 Similaridade de Strings
- **Distância de Levenshtein**: Edições mínimas
- **Jaro-Winkler**: Para nomes próprios
- **Soundex**: Similaridade fonética

### 4. Análise de Sentimentos

#### 4.1 Abordagens Lexicais
Baseadas em dicionários de palavras com polaridade.

**Lexicons populares:**
- **VADER**: Valence Aware Dictionary and sEntiment Reasoner
- **SentiWordNet**: WordNet com scores de sentimento
- **AFINN**: Lista de palavras com scores -5 a +5
- **TextBlob**: Implementação simples

#### 4.2 Características dos Sentimentos
- **Polaridade**: Positivo, negativo, neutro
- **Intensidade**: Grau do sentimento (0.0 a 1.0)
- **Subjetividade**: Objetivo vs subjetivo
- **Emoções**: Alegria, raiva, medo, tristeza, etc.

#### 4.3 Desafios
- **Ironia e sarcasmo**
- **Contexto dependente**
- **Negação**: "não é bom" vs "é bom"
- **Intensificadores**: "muito bom" vs "bom"

### 5. Visualização de Dados Textuais

#### 5.1 Gráficos de Frequência
- **Histogramas**: Distribuição de comprimentos
- **Gráficos de barras**: Top palavras mais frequentes
- **Curvas de Zipf**: Lei de potência
- **Box plots**: Estatísticas descritivas

#### 5.2 Nuvens de Palavras
- **Word clouds**: Visualização intuitiva
- **Configurações**: Cores, fontes, layouts
- **Máscaras**: Formas personalizadas
- **Filtros**: Stopwords, frequência mínima

#### 5.3 Visualizações Avançadas
- **Heatmaps**: Matrizes de correlação
- **Scatter plots**: Relações entre métricas
- **Time series**: Evolução temporal
- **Network graphs**: Relações entre entidades

### 6. Análise de Corpus

#### 6.1 Comparação de Corpus
- **Diferenças vocabulares**
- **Palavras distintivas**
- **Análise temporal**
- **Segmentação por domínio**

#### 6.2 Detecção de Tópicos (Básica)
- **TF-IDF para tópicos**
- **Clustering de documentos**
- **Análise de co-ocorrência**
- **Preparação para LDA**

#### 6.3 Métricas de Qualidade
- **Diversidade lexical**: Type-Token Ratio (TTR)
- **Complexidade**: Comprimento médio de sentenças
- **Legibilidade**: Índices de Flesch, ARI
- **Coerência**: Medidas de coesão textual

## 🛠 Ferramentas e Implementações

### Bibliotecas Principais
```python
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.collocations import BigramCollocationFinder
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
```

### Análise de N-gramas
```python
from nltk import ngrams
from nltk.util import ngrams as nltk_ngrams

def extrair_ngrams(texto, n=2):
    tokens = word_tokenize(texto.lower())
    return list(ngrams(tokens, n))

def contar_ngrams(textos, n=2, top_k=20):
    todos_ngrams = []
    for texto in textos:
        todos_ngrams.extend(extrair_ngrams(texto, n))
    
    counter = Counter(todos_ngrams)
    return counter.most_common(top_k)
```

### Medidas de Similaridade
```python
def similaridade_jaccard(set1, set2):
    intersecao = len(set1.intersection(set2))
    uniao = len(set1.union(set2))
    return intersecao / uniao if uniao > 0 else 0

def similaridade_coseno(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norma1 = np.linalg.norm(vec1)
    norma2 = np.linalg.norm(vec2)
    return dot_product / (norma1 * norma2) if norma1 > 0 and norma2 > 0 else 0
```

## 📊 Exercícios Práticos

### Exercício 1: Análise de Frequência
- Calcular distribuição de palavras
- Verificar se segue a Lei de Zipf
- Identificar palavras características por categoria

### Exercício 2: N-gramas
- Extrair bigramas e trigramas mais frequentes
- Encontrar colocações significativas
- Comparar n-gramas entre diferentes textos

### Exercício 3: Similaridade de Textos
- Implementar diferentes métricas de similaridade
- Criar matriz de similaridade entre documentos
- Encontrar textos mais similares

### Exercício 4: Análise de Sentimentos
- Aplicar diferentes métodos de análise
- Comparar resultados entre abordagens
- Analisar evolução temporal de sentimentos

### Exercício 5: Visualizações
- Criar nuvens de palavras temáticas
- Plotar distribuições estatísticas
- Desenvolver dashboard interativo

## 📈 Métricas e Avaliação

### Métricas de Corpus
- **Type-Token Ratio (TTR)**: Diversidade lexical
- **Mean Length of Utterance (MLU)**: Complexidade
- **Hapax Ratio**: Proporção de palavras únicas
- **Entropy**: Medida de incerteza

### Validação de Resultados
- **Inspeção manual**: Verificar top resultados
- **Comparação com baselines**: Métodos simples
- **Correlação humana**: Acordo com anotadores
- **Robustez**: Estabilidade com diferentes dados

## 🚨 Armadilhas Comuns

### Problemas Estatísticos
- **Confundir frequência com importância**
- **Ignorar normalização por comprimento**
- **Não considerar distribuição de corpus**
- **Usar médias quando distribuição é skewed**

### Problemas de Interpretação
- **Assumir causalidade de correlação**
- **Generalizar de amostras pequenas**
- **Ignorar contexto linguístico**
- **Não validar com especialistas**

## 📖 Leituras Complementares

### Livros
- "Foundations of Statistical Natural Language Processing" - Manning & Schütze
- "Text Analysis with R for Students of Literature" - Jockers
- "Quantitative Corpus Linguistics with R" - Gries

### Papers
- "Zipf's Word Frequency Law in Natural Language" - Piantadosi (2014)
- "The Mathematics of Statistical Machine Translation" - Brown et al. (1993)
- "Pointwise Mutual Information" - Church & Hanks (1990)

## 🎯 Próximos Passos

No **Módulo 4**, vamos aprender sobre representação vetorial de texto, incluindo Bag of Words, TF-IDF e word embeddings.

---

**Dica**: Execute o notebook `03_analise_estatistica.ipynb` para aplicar todos os conceitos na prática! 