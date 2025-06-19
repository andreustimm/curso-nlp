[⬅️ Voltar para o índice do curso](../README.md)

# Módulo 5: Classificação de Texto

## 🎯 Objetivos do Módulo

Ao final deste módulo, você será capaz de:
- Implementar classificadores Naive Bayes para texto
- Aplicar SVM e regressão logística em NLP
- Usar ensemble methods para melhorar performance
- Realizar validação cruzada e avaliar modelos
- Lidar com dados desbalanceados
- Otimizar hiperparâmetros de classificadores

## 📚 Conteúdo Teórico

### 1. Fundamentos da Classificação de Texto

#### 1.1 Definição da Tarefa
Classificação de texto é a tarefa de atribuir categorias predefinidas a documentos baseado em seu conteúdo.

**Tipos de Classificação:**
- **Binária**: Duas classes (spam/não-spam)
- **Multi-classe**: Múltiplas classes mutuamente exclusivas
- **Multi-label**: Múltiplas classes não exclusivas
- **Hierárquica**: Classes organizadas em taxonomia

#### 1.2 Pipeline de Classificação
1. **Coleta e rotulação** de dados
2. **Pré-processamento** de texto
3. **Extração de features** (representação)
4. **Divisão** treino/validação/teste
5. **Treinamento** do modelo
6. **Avaliação** e otimização
7. **Deploy** e monitoramento

### 2. Naive Bayes para Texto

#### 2.1 Teorema de Bayes
P(classe|documento) = P(documento|classe) × P(classe) / P(documento)

**Componentes:**
- **P(classe)**: Probabilidade a priori
- **P(documento|classe)**: Verossimilhança
- **P(documento)**: Evidência (constante)

#### 2.2 Assumption de Independência
"Naive" porque assume independência condicional entre features.

**Na prática:**
- Palavras são tratadas independentemente
- Simplificação forte mas funciona bem
- Reduz complexidade computacional

#### 2.3 Variantes do Naive Bayes

**Multinomial NB:**
- Para contagens de palavras
- Boa para classificação de tópicos
- Assume distribuição multinomial

**Bernoulli NB:**
- Para presença/ausência de palavras
- Bom para textos curtos
- Features binárias

**Gaussian NB:**
- Para features contínuas
- Assume distribuição normal
- Útil com embeddings

#### 2.4 Suavização (Smoothing)
Evita probabilidades zero para palavras não vistas.

**Add-one (Laplace) Smoothing:**
P(palavra|classe) = (count + 1) / (total + vocab_size)

**Add-alpha Smoothing:**
P(palavra|classe) = (count + α) / (total + α × vocab_size)

### 3. Support Vector Machines (SVM)

#### 3.1 Conceito Básico
SVM encontra hiperplano que separa classes com margem máxima.

**Vantagens para texto:**
- Eficaz em alta dimensionalidade
- Funciona bem com dados esparsos
- Resistente a overfitting
- Suporte a kernels não-lineares

#### 3.2 Kernels para Texto

**Linear Kernel:**
- K(x,y) = x·y
- Mais comum para texto
- Interpretável e eficiente

**RBF (Gaussian) Kernel:**
- K(x,y) = exp(-γ||x-y||²)
- Para relações não-lineares
- Requer normalização

**Polynomial Kernel:**
- K(x,y) = (x·y + c)^d
- Captura interações entre features
- Cuidado com overfitting

#### 3.3 Hiperparâmetros
- **C**: Regularização (trade-off bias-variance)
- **gamma**: Para kernels RBF
- **degree**: Para kernels polinomiais

### 4. Regressão Logística

#### 4.1 Modelo Linear Generalizado
Usa função logística para mapear scores para probabilidades.

**Função Sigmoide:**
P(y=1|x) = 1 / (1 + e^(-wᵀx))

#### 4.2 Vantagens
- Retorna probabilidades calibradas
- Interpretável (coeficientes = importância)
- Baseline forte para muitas tarefas
- Treinamento eficiente

#### 4.3 Regularização

**L1 (Lasso):**
- Penalty: λ∑|wᵢ|
- Feature selection automática
- Produz esparsidade

**L2 (Ridge):**
- Penalty: λ∑wᵢ²
- Previne overfitting
- Mantém todas as features

**Elastic Net:**
- Combina L1 e L2
- α∑|wᵢ| + (1-α)∑wᵢ²

### 5. Ensemble Methods

#### 5.1 Voting Classifiers

**Hard Voting:**
- Voto majoritário
- Cada modelo contribui 1 voto

**Soft Voting:**
- Média das probabilidades
- Usa confiança dos modelos

#### 5.2 Bagging

**Random Forest:**
- Múltiplas árvores
- Bootstrap sampling
- Feature randomness

**Extra Trees:**
- Árvores completamente aleatórias
- Splits aleatórios
- Menos overfitting

#### 5.3 Boosting

**AdaBoost:**
- Foca em exemplos difíceis
- Pesos adaptativos
- Combina weak learners

**Gradient Boosting:**
- XGBoost, LightGBM, CatBoost
- Otimização sequencial
- Estado da arte em tabular

### 6. Avaliação de Modelos

#### 6.1 Métricas para Classificação

**Accuracy:**
- (TP + TN) / (TP + TN + FP + FN)
- Útil quando classes balanceadas

**Precision:**
- TP / (TP + FP)
- "Dos que classifiquei como positivos, quantos realmente são?"

**Recall (Sensitivity):**
- TP / (TP + FN)
- "Dos positivos reais, quantos consegui identificar?"

**F1-Score:**
- 2 × (Precision × Recall) / (Precision + Recall)
- Média harmônica de precision e recall

#### 6.2 Métricas Multi-classe

**Macro Average:**
- Calcula métrica para cada classe
- Média não ponderada
- Trata classes igualmente

**Micro Average:**
- Agrega TPs, FPs, FNs globalmente
- Calculado uma única métrica
- Favorece classes majoritárias

**Weighted Average:**
- Média ponderada por suporte
- Considera distribuição de classes

#### 6.3 Validação Cruzada

**K-Fold:**
- Divide dados em k partições
- k rounds de treino/validação
- Média dos resultados

**Stratified K-Fold:**
- Mantém proporção de classes
- Importante para dados desbalanceados

**Time Series Split:**
- Para dados temporais
- Treino sempre anterior à validação

### 7. Dados Desbalanceados

#### 7.1 Problemas
- Accuracy enganosa
- Bias para classe majoritária
- Recall baixo para classe minoritária

#### 7.2 Técnicas de Sampling

**Under-sampling:**
- Remove exemplos da classe majoritária
- SMOTE, Tomek Links, EditedNearestNeighbours

**Over-sampling:**
- Adiciona exemplos da classe minoritária
- SMOTE, ADASYN, BorderlineSMOTE

**Hybrid:**
- Combina under e over-sampling
- SMOTEENN, SMOTETomek

#### 7.3 Técnicas Algorítmicas

**Class Weights:**
- Penaliza erros em classes minoritárias
- class_weight='balanced'

**Threshold Adjustment:**
- Ajusta threshold de decisão
- Otimiza para métrica específica

**Cost-Sensitive Learning:**
- Matrix de custos customizada
- Penaliza diferentes tipos de erro

## 🛠 Implementações Práticas

### Pipeline Completo
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', LogisticRegression())
])

parameters = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'classifier__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='f1_macro')
```

### Naive Bayes
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
classifier = MultinomialNB(alpha=1.0)
classifier.fit(X, y)
```

### SVM
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Para embeddings (features densas)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dense)
svm = SVC(kernel='rbf', C=1.0, probability=True)
svm.fit(X_scaled, y)
```

## 📊 Exercícios Práticos

### Exercício 1: Classificação de Sentimentos
- Dataset de reviews de produtos
- Comparar Naive Bayes, SVM, Logistic Regression
- Análise de features mais importantes

### Exercício 2: Classificação de Notícias
- Dataset multi-classe
- Ensemble de diferentes algoritmos
- Otimização de hiperparâmetros

### Exercício 3: Detecção de Spam
- Dataset desbalanceado
- Técnicas para dados desbalanceados
- Análise de falsos positivos/negativos

### Exercício 4: Pipeline Completo
- From raw text to predictions
- Cross-validation robusta
- Feature engineering avançado

## 🎯 Próximos Passos

No **Módulo 6**, exploraremos modelos de sequência como HMMs e CRFs para tarefas estruturadas.

---

**Dica**: Execute o notebook `05_classificacao_texto.ipynb` para implementar todos os algoritmos! 

[⬅️ Voltar para o índice do curso](../README.md) 