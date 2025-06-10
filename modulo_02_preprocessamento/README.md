# Módulo 2: Pré-processamento de Texto

## 🎯 Objetivos do Módulo

Ao final deste módulo, você será capaz de:
- Compreender a importância do pré-processamento em NLP
- Implementar técnicas de limpeza e normalização de texto
- Aplicar tokenização avançada
- Utilizar stemming e lemmatização eficientemente
- Trabalhar com expressões regulares para processamento de texto
- Remover stopwords de forma inteligente

## 📚 Conteúdo Teórico

### 1. Importância do Pré-processamento

O pré-processamento é uma etapa crucial em qualquer pipeline de NLP, pois:

- **Reduz o ruído**: Remove caracteres irrelevantes e padroniza o texto
- **Normaliza dados**: Garante consistência no formato dos dados
- **Melhora performance**: Reduz dimensionalidade e complexidade
- **Facilita análise**: Prepara texto para algoritmos de ML/DL

### 2. Etapas do Pré-processamento

#### 2.1 Limpeza de Texto

**Remoção de elementos indesejados:**
- HTML tags: `<div>`, `<p>`, `<br>`
- URLs: `http://exemplo.com`
- Menções: `@usuario`
- Hashtags: `#nlp`
- Emojis e caracteres especiais
- Números (quando irrelevantes)

**Normalização de caracteres:**
- Conversão para minúsculas/maiúsculas
- Remoção de acentos
- Normalização de espaços em branco
- Correção de encoding

#### 2.2 Tokenização

**Definição**: Processo de dividir texto em unidades menores (tokens)

**Tipos de tokenização:**
- **Por palavras**: Mais comum
- **Por sentenças**: Para análise sintática
- **Por subpalavras**: BPE, WordPiece
- **Por caracteres**: Para idiomas sem espaços

**Desafios:**
- Contrações: "não é" vs "não" + "é"
- Pontuação: "Dr. Silva" vs "Dr." + "Silva"
- URLs e emails
- Números com formatação

#### 2.3 Normalização de Texto

**Case folding:**
- Conversão para minúsculas
- Preservação de acrônimos quando necessário

**Expansão de contrações:**
- "don't" → "do not"
- "I'm" → "I am"
- "won't" → "will not"

**Padronização de abreviações:**
- "Dr." → "Doctor"
- "etc." → "et cetera"

#### 2.4 Remoção de Stopwords

**Definição**: Palavras muito frequentes que geralmente não carregam significado específico

**Stopwords comuns em português:**
- Artigos: a, o, as, os
- Preposições: de, para, com, em
- Pronomes: eu, você, ele, ela
- Verbos auxiliares: ser, estar, ter

**Considerações:**
- Dependem do domínio e tarefa
- Podem ser importantes em algumas análises
- Listas personalizadas podem ser necessárias

#### 2.5 Stemming vs Lemmatização

**Stemming:**
- Remove sufixos usando regras
- Mais rápido, menos preciso
- Pode gerar palavras inexistentes
- Exemplo: "correndo" → "corr"

**Lemmatização:**
- Reduz palavras à forma canônica
- Mais lento, mais preciso
- Considera contexto gramatical
- Exemplo: "correndo" → "correr"

**Quando usar cada um:**
- **Stemming**: Tarefas de recuperação de informação
- **Lemmatização**: Análise sintática e semântica

### 3. Ferramentas e Bibliotecas

#### 3.1 NLTK
```python
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
```

#### 3.2 spaCy
```python
import spacy
nlp = spacy.load("pt_core_news_sm")
doc = nlp("texto")
tokens = [token.lemma_ for token in doc]
```

#### 3.3 Expressões Regulares
```python
import re
# Remover URLs
texto = re.sub(r'http\S+', '', texto)
# Remover menções
texto = re.sub(r'@\w+', '', texto)
```

### 4. Técnicas Avançadas

#### 4.1 Normalização Unicode
- **NFD**: Decomposição canônica
- **NFC**: Composição canônica
- **NFKD/NFKC**: Normalização de compatibilidade

#### 4.2 Detecção de Idioma
- Identificar automaticamente o idioma do texto
- Aplicar pré-processamento específico por idioma

#### 4.3 Correção Ortográfica
- Detecção de erros de digitação
- Sugestão de correções
- Normalização de variações

#### 4.4 Segmentação de Sentenças
- Divisão correta em sentenças
- Tratamento de abreviações
- Pontuação especial

### 5. Boas Práticas

#### 5.1 Pipeline Consistente
- Aplicar sempre as mesmas transformações
- Documentar todas as etapas
- Manter versionamento do pipeline

#### 5.2 Preservação de Informação
- Manter texto original quando possível
- Registrar transformações aplicadas
- Permitir reversibilidade quando necessário

#### 5.3 Validação
- Verificar resultados em amostra
- Testar com casos extremos
- Monitorar impacto na performance

#### 5.4 Eficiência
- Paralelizar quando possível
- Usar bibliotecas otimizadas
- Cache de resultados frequentes

## 🛠 Ferramentas Específicas

### Expressões Regulares Úteis

```python
# URLs
r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# E-mails
r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

# Telefones (Brasil)
r'\(?\d{2}\)?\s?\d{4,5}-?\d{4}'

# Datas
r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'

# Hashtags
r'#\w+'

# Menções
r'@\w+'
```

### Stopwords Customizadas

```python
# Stopwords básicas + domínio específico
stopwords_custom = stopwords.words('portuguese') + [
    'site', 'página', 'clique', 'aqui', 'mais',
    'muito', 'bem', 'bom', 'boa', 'melhor'
]
```

## 📊 Métricas de Qualidade

### Antes vs Depois do Pré-processamento
- **Vocabulário**: Redução do número de tokens únicos
- **Esparsidade**: Diminuição da matriz de características
- **Consistência**: Padronização de variações
- **Relevância**: Manutenção de informação importante

### Validação Manual
- Amostragem de resultados
- Verificação de casos extremos
- Revisão por especialistas
- Testes A/B com diferentes pipelines

## 💡 Exercícios Práticos

1. **Limpeza de Tweets**: Processar dados do Twitter
2. **Normalização de Reviews**: Padronizar avaliações de produtos
3. **Pipeline Personalizado**: Criar pipeline para domínio específico
4. **Comparação de Métodos**: Stemming vs Lemmatização
5. **Regex Avançado**: Extrair informações específicas

## 🚨 Armadilhas Comuns

### Over-processing
- Remover informação relevante
- Aplicar transformações desnecessárias
- Perder contexto importante

### Under-processing
- Manter ruído desnecessário
- Não normalizar variações
- Ignorar características específicas do domínio

### Inconsistência
- Aplicar regras diferentes em treino/teste
- Mudanças no pipeline durante o projeto
- Não documentar transformações

## 📖 Leituras Complementares

### Artigos
- "Text Preprocessing in Python: Steps, Tools, and Examples"
- "The Art of Cleaning Your Data"
- "Regular Expressions for Text Processing"

### Documentação
- [NLTK Preprocessing](https://www.nltk.org/book/ch03.html)
- [spaCy Linguistic Features](https://spacy.io/usage/linguistic-features)
- [Regex Python Documentation](https://docs.python.org/3/library/re.html)

## 🎯 Próximos Passos

No **Módulo 3**, vamos explorar análise estatística de texto, incluindo análise de frequência, n-gramas e medidas de similaridade.

---

**Dica**: Execute o notebook `02_preprocessamento_texto.ipynb` para praticar todas as técnicas! 