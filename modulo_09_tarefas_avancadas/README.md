[⬅️ Voltar](../README.md)

# Módulo 9: Tarefas Avançadas de NLP

## 🎯 Objetivos do Módulo

Ao final deste módulo, você será capaz de:
- Implementar sistemas de Question Answering
- Construir modelos de sumarização automática
- Desenvolver sistemas de tradução neural
- Aplicar modelos em análise de diálogo
- Implementar detecção de tópicos com LDA
- Criar sistemas de recomendação baseados em texto

## 📚 Conteúdo Teórico

### 1. Question Answering (QA)

#### 1.1 Tipos de QA Systems

**Extractive QA:**
- Resposta é span do texto
- SQuAD, MS MARCO
- BERT-based models

**Generative QA:**
- Resposta é gerada
- Pode sintetizar informação
- T5, BART, GPT

**Open-domain QA:**
- Busca em grande corpus
- Retrieve + Read pipeline
- DPR (Dense Passage Retrieval)

#### 1.2 Arquiteturas para QA

**BERT for QA:**
```
[CLS] question [SEP] context [SEP]
↓
BERT Encoder
↓
Start/End Position Prediction
```

**T5 for QA:**
```
Input: "question: What is NLP? context: NLP is..."
Output: "Natural Language Processing"
```

#### 1.3 Metrics de Avaliação
- **Exact Match (EM)**: Resposta exata
- **F1 Score**: Overlap de tokens
- **BLEU**: Para respostas generativas
- **ROUGE**: Para sumarização

### 2. Sumarização Automática

#### 2.1 Tipos de Sumarização

**Extractive:**
- Seleciona sentenças importantes
- Preserva texto original
- TextRank, BERT-based

**Abstractive:**
- Gera novo texto
- Pode parafrasear
- Seq2Seq, Transformer

**Single vs Multi-document:**
- Um documento vs múltiplos
- Complexidade de fusão
- Redundância e contradições

#### 2.2 Algoritmos Extractivos

**TextRank:**
- PageRank para sentenças
- Grafo de similaridade
- Não supervisionado

**BERT-based:**
- BERT embeddings
- Clustering ou ranking
- Fine-tuning possible

#### 2.3 Modelos Abstractivos

**BART:**
- Denoising autoencoder
- Encoder-decoder
- State-of-the-art performance

**T5:**
- Text-to-Text Transfer Transformer
- "summarize: [text]"
- Unified framework

**PEGASUS:**
- Pré-treinado especificamente para sumarização
- Gap sentence generation
- Strong baseline

### 3. Tradução Neural (NMT)

#### 3.1 Evolução da Tradução

**Statistical MT (SMT):**
- Phrase-based translation
- Language models
- Alignment models

**Neural MT (NMT):**
- End-to-end neural networks
- Better fluency
- Context awareness

#### 3.2 Arquiteturas NMT

**Seq2Seq with Attention:**
- Encoder-Decoder RNN/LSTM
- Attention mechanism
- Alinhamento automático

**Transformer NMT:**
- Self-attention
- Paralelização
- State-of-the-art

#### 3.3 Desafios em NMT
- **Low-resource languages**
- **Domain adaptation**
- **Rare words (OOV)**
- **Long sentences**
- **Evaluation metrics**

### 4. Topic Modeling

#### 4.1 Latent Dirichlet Allocation (LDA)

**Intuição:**
- Documentos são misturas de tópicos
- Tópicos são distribuições de palavras
- Inferência Bayesiana

**Modelo Generativo:**
```
Para cada documento d:
  θ_d ~ Dirichlet(α)  # Distribuição de tópicos
  Para cada palavra w:
    z ~ Multinomial(θ_d)  # Escolher tópico
    w ~ Multinomial(φ_z)  # Gerar palavra
```

#### 4.2 Algoritmos de Inferência

**Gibbs Sampling:**
- MCMC method
- Sampling z_i dado z_{-i}
- Convergência lenta mas precisa

**Variational Inference:**
- Aproximação determinística
- Mais rápido que Gibbs
- Usado no scikit-learn

#### 4.3 Avaliação de Tópicos
- **Perplexity**: Likelihood dos dados
- **Coherence**: Coerência semântica
- **Topic diversity**: Variedade de tópicos
- **Human evaluation**: Julgamento humano

### 5. Análise de Diálogo

#### 5.1 Componentes de Chatbots

**Natural Language Understanding (NLU):**
- Intent classification
- Entity extraction
- Slot filling

**Dialogue Management:**
- State tracking
- Policy learning
- Context maintenance

**Natural Language Generation (NLG):**
- Response generation
- Template-based vs neural
- Personality and style

#### 5.2 Intent Classification
```
User: "Book a flight to Paris"
Intent: book_flight
Entities: {destination: "Paris"}
```

#### 5.3 Response Generation

**Template-based:**
- Rule-based responses
- Slot filling
- Deterministic

**Neural Generation:**
- Seq2Seq models
- Personality modeling
- Contextual responses

### 6. Information Extraction

#### 6.1 Named Entity Recognition (NER)

**Nested NER:**
- Entidades aninhadas
- "Apple Inc. CEO Tim Cook"
- Estruturas complexas

**Few-shot NER:**
- Poucos exemplos
- Meta-learning
- Transfer learning

#### 6.2 Relation Extraction
- Identificar relações entre entidades
- "Tim Cook works for Apple"
- Distant supervision

#### 6.3 Event Extraction
- Detectar eventos em texto
- Participantes, tempo, local
- ACE datasets

## 🛠 Implementações Práticas

### Question Answering com BERT
```python
from transformers import BertForQuestionAnswering, BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors='pt', 
                      truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx+1])
    
    return answer
```

### Sumarização com BART
```python
from transformers import BartTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def summarize_text(text, max_length=150):
    inputs = tokenizer(text, return_tensors='pt', 
                      max_length=1024, truncation=True)
    
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

### Topic Modeling com LDA
```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Preparar dados
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(documents)

# Treinar LDA
lda = LatentDirichletAllocation(
    n_components=10,
    random_state=42,
    max_iter=100
)
lda.fit(doc_term_matrix)

# Extrair tópicos
feature_names = vectorizer.get_feature_names_out()

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-no_top_words:]]
        print(f"Topic {topic_idx}: {' '.join(top_words)}")
```

### Tradução com MarianMT
```python
from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-pt'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)
```

## 📊 Exercícios Práticos

### Exercício 1: Sistema de QA
- Implementar QA extractivo e generativo
- Avaliar em dataset português
- Interface web simples

### Exercício 2: Sumarizador de Notícias
- Comparar métodos extractivos e abstractivos
- Avaliar com ROUGE
- Pipeline end-to-end

### Exercício 3: Análise de Tópicos
- LDA em corpus de documentos
- Visualização de tópicos
- Evolução temporal

### Exercício 4: Chatbot Simples
- Intent classification
- Entity extraction
- Response generation

### Exercício 5: Sistema de Tradução
- Fine-tuning para domínio específico
- Avaliação automática e humana
- Análise de erros

## 🎯 Próximos Passos

No **Módulo 10**, integraremos tudo em projetos práticos completos para portfólio.

---

**Dica**: Execute o notebook `09_tarefas_avancadas_nlp.ipynb` para implementar sistemas completos! 

[⬅️ Voltar](../README.md) 