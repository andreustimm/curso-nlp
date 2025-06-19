[⬅️ Voltar para o índice do curso](../README.md)

# Módulo 1: Fundamentos de NLP

## 🎯 Objetivos do Módulo

Ao final deste módulo, você será capaz de:
- Compreender o que é Processamento de Linguagem Natural
- Conhecer a história e evolução do NLP
- Identificar as principais aplicações do NLP
- Entender os desafios fundamentais do processamento de linguagem
- Conhecer o pipeline básico de um sistema de NLP

## 📚 Conteúdo Teórico

### 1. O que é NLP?

O **Processamento de Linguagem Natural (NLP)** é uma subárea da Inteligência Artificial que combina ciência da computação, linguística e aprendizado de máquina para permitir que computadores compreendam, interpretem e gerem linguagem humana de forma útil.

### 2. História do NLP

#### Década de 1950-1960: Primórdios
- **1950**: Teste de Turing
- **1954**: Primeiro experimento de tradução automática (Georgetown-IBM)
- **1960s**: Sistemas baseados em regras

#### Década de 1980-1990: Era Estatística
- **1980s**: Introdução de métodos estatísticos
- **1990s**: Corpus linguistics e modelos probabilísticos

#### Década de 2000-2010: Machine Learning
- **2000s**: SVM, Naive Bayes para NLP
- **2003**: Word embeddings iniciais
- **2006**: Deep learning ressurge

#### Década de 2010-presente: Deep Learning
- **2013**: Word2Vec
- **2017**: Transformer (Attention is All You Need)
- **2018**: BERT revoluciona o campo
- **2019-2023**: Era dos Large Language Models (GPT, ChatGPT)

### 3. Principais Aplicações

#### Comunicação e Tradução
- **Tradução Automática**: Google Translate, DeepL
- **Correção Ortográfica**: Grammarly, corretor do Word
- **Assistentes Virtuais**: Siri, Alexa, Google Assistant

#### Análise de Texto
- **Análise de Sentimentos**: Monitoramento de redes sociais
- **Classificação de Documentos**: Organização automática
- **Extração de Informações**: Named Entity Recognition

#### Geração de Conteúdo
- **Chatbots**: Atendimento ao cliente
- **Sumarização**: Resumo automático de textos
- **Geração de Texto**: GPT, Claude, ChatGPT

#### Busca e Recuperação
- **Motores de Busca**: Google, Bing
- **Sistemas de Recomendação**: Netflix, Amazon
- **Question Answering**: Sistemas de perguntas e respostas

### 4. Desafios Fundamentais do NLP

#### Ambiguidade
- **Lexical**: Uma palavra com múltiplos significados
  - "Banco" → instituição financeira ou assento
- **Sintática**: Estrutura gramatical ambígua
  - "Vi o homem com o telescópio"
- **Semântica**: Significado ambíguo
  - "Ele foi ao banco" (contexto determina o significado)

#### Variabilidade Linguística
- **Dialetos e Sotaques**: Variações regionais
- **Gírias e Neologismos**: Linguagem em constante evolução
- **Linguagem Informal**: Redes sociais, mensagens

#### Contexto e Pragmática
- **Referência**: "Ele chegou cedo" (quem é "ele"?)
- **Ironia e Sarcasmo**: "Que dia lindo!" (em dia chuvoso)
- **Conhecimento de Mundo**: Informações implícitas

#### Diversidade de Idiomas
- **Morfologia Rica**: Português, alemão
- **Ordem de Palavras**: SOV vs SVO vs VSO
- **Sistemas de Escrita**: Latino, árabe, chinês

### 5. Pipeline Básico de NLP

```
Texto Bruto → Pré-processamento → Análise → Modelagem → Aplicação
```

#### Etapa 1: Pré-processamento
- **Limpeza**: Remoção de caracteres especiais
- **Tokenização**: Divisão em palavras/tokens
- **Normalização**: Padronização do texto

#### Etapa 2: Análise Linguística
- **Análise Lexical**: Identificação de palavras
- **Análise Sintática**: Estrutura gramatical
- **Análise Semântica**: Significado das palavras

#### Etapa 3: Representação
- **Vetorização**: Conversão para números
- **Embeddings**: Representações densas
- **Features**: Características relevantes

#### Etapa 4: Modelagem
- **Algoritmos de ML**: Classificação, clustering
- **Deep Learning**: Redes neurais
- **Pré-treinados**: BERT, GPT

#### Etapa 5: Aplicação
- **Interface**: APIs, aplicações web
- **Avaliação**: Métricas de performance
- **Deploy**: Produção e monitoramento

## 🛠 Ferramentas e Bibliotecas

### Python Libraries
- **NLTK**: Natural Language Toolkit
- **spaCy**: Industrial-strength NLP
- **Gensim**: Topic modeling
- **TextBlob**: Simplified text processing

### Deep Learning
- **TensorFlow/Keras**: Google's framework
- **PyTorch**: Facebook's framework
- **Transformers**: Hugging Face

### Visualização
- **WordCloud**: Nuvens de palavras
- **Matplotlib/Seaborn**: Gráficos
- **Plotly**: Visualizações interativas

## 📊 Métricas de Avaliação

### Classificação
- **Accuracy**: Taxa de acerto geral
- **Precision**: Precisão por classe
- **Recall**: Revocação por classe
- **F1-Score**: Média harmônica de precision e recall

### Geração de Texto
- **BLEU**: Bilingual Evaluation Understudy
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
- **Perplexity**: Medida de incerteza do modelo

### Embeddings
- **Cosine Similarity**: Similaridade entre vetores
- **Analogies**: Capacidade de resolver analogias
- **Clustering**: Qualidade dos agrupamentos

## 💡 Exercícios Práticos

1. **Exploração de Corpus**: Analise um conjunto de textos
2. **Pipeline Simples**: Implemente um pipeline básico
3. **Comparação de Ferramentas**: Compare NLTK vs spaCy
4. **Análise Exploratória**: Visualize características de texto

## 📖 Leituras Complementares

### Livros
- "Speech and Language Processing" - Jurafsky & Martin
- "Natural Language Processing with Python" - Steven Bird
- "Introduction to Information Retrieval" - Manning et al.

### Papers Fundamentais
- "Attention Is All You Need" (Transformer)
- "BERT: Pre-training of Deep Bidirectional Transformers"
- "Efficient Estimation of Word Representations in Vector Space" (Word2Vec)

### Recursos Online
- [NLP Course by Hugging Face](https://huggingface.co/course/)
- [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [spaCy Documentation](https://spacy.io/)

## 🎯 Próximos Passos

No **Módulo 2**, vamos mergulhar no pré-processamento de texto, aprendendo técnicas essenciais como tokenização, normalização, stemming e lemmatização.

---

**Dica**: Execute o notebook `01_fundamentos_nlp.ipynb` para ver os conceitos na prática!

[⬅️ Voltar para o índice do curso](../README.md) 