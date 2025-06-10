# Guia de Instalação - Curso de NLP

## 🚀 Instalação Rápida (Recomendada)

### Opção 1: Script Automático
```bash
# Clone o repositório
git clone <url-do-repositorio>
cd curso-nlp

# Execute o script de configuração
python setup_curso.py
```

### Opção 2: Instalação Manual
```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Baixar modelo do spaCy em português
python -m spacy download pt_core_news_sm

# 3. Baixar recursos do NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 4. Iniciar Jupyter
jupyter notebook
```

## 📋 Pré-requisitos

### Sistema Operacional
- **Windows 10+**, **macOS 10.14+**, ou **Linux Ubuntu 18.04+**
- **Python 3.8 ou superior**
- **8GB RAM** (recomendado: 16GB)
- **5GB espaço livre** em disco

### Conhecimentos Prévios
- **Python básico**: variáveis, funções, classes
- **Pandas/NumPy**: manipulação de dados
- **Matplotlib**: visualização básica
- **Machine Learning**: conceitos fundamentais (opcional)

## 🐍 Configuração do Ambiente Python

### Opção 1: Anaconda (Recomendada)
```bash
# Baixar Anaconda: https://www.anaconda.com/products/distribution

# Criar ambiente virtual
conda create -n nlp-curso python=3.9
conda activate nlp-curso

# Instalar dependências
pip install -r requirements.txt
```

### Opção 2: venv (Python padrão)
```bash
# Criar ambiente virtual
python -m venv nlp-curso

# Ativar ambiente (Windows)
nlp-curso\Scripts\activate

# Ativar ambiente (macOS/Linux)
source nlp-curso/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### Opção 3: Poetry (Avançado)
```bash
# Instalar Poetry: https://python-poetry.org/docs/#installation

# Instalar dependências
poetry install

# Ativar ambiente
poetry shell
```

## 📦 Dependências Principais

### Bibliotecas de NLP
```bash
# NLTK - Natural Language Toolkit
pip install nltk==3.8.1

# spaCy - Industrial-strength NLP
pip install spacy==3.4.4
python -m spacy download pt_core_news_sm

# Transformers - Estado da arte
pip install transformers==4.25.1

# Gensim - Topic modeling
pip install gensim==4.2.0
```

### Machine Learning
```bash
# Scikit-learn
pip install scikit-learn==1.2.0

# TensorFlow
pip install tensorflow==2.11.0

# PyTorch (opcional)
pip install torch==1.13.1
```

### Visualização e Análise
```bash
# Jupyter e extensões
pip install jupyter==1.0.0
pip install ipywidgets==8.0.4

# Visualização
pip install matplotlib==3.6.2
pip install seaborn==0.12.1
pip install plotly==5.11.0
pip install wordcloud==1.9.2
```

## 🔧 Configuração Específica por Sistema

### Windows
```bash
# Instalar Microsoft C++ Build Tools se necessário
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Configurar encoding UTF-8
set PYTHONIOENCODING=utf-8
```

### macOS
```bash
# Instalar Xcode Command Line Tools
xcode-select --install

# Configurar locale (se necessário)
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

### Linux (Ubuntu/Debian)
```bash
# Instalar dependências do sistema
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo apt-get install build-essential

# Para spaCy
sudo apt-get install python3-dev
```

## 🧪 Verificação da Instalação

### Teste Rápido
```python
# Execute este código para verificar se tudo está funcionando
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import spacy
from transformers import pipeline

print("✅ Todas as bibliotecas importadas com sucesso!")

# Testar spaCy
nlp = spacy.load("pt_core_news_sm")
doc = nlp("Olá mundo!")
print(f"✅ spaCy funcionando: {[token.text for token in doc]}")

# Testar transformers
classifier = pipeline("sentiment-analysis")
result = classifier("I love this course!")
print(f"✅ Transformers funcionando: {result}")
```

### Script de Diagnóstico
```bash
# Execute o script de diagnóstico
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import nltk
    print(f'✅ NLTK: {nltk.__version__}')
except ImportError:
    print('❌ NLTK não encontrado')

try:
    import spacy
    print(f'✅ spaCy: {spacy.__version__}')
    nlp = spacy.load('pt_core_news_sm')
    print('✅ Modelo português carregado')
except:
    print('❌ Problema com spaCy ou modelo')

try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except ImportError:
    print('❌ Transformers não encontrado')
"
```

## 🚨 Solução de Problemas Comuns

### Erro: "No module named 'xxx'"
```bash
# Verificar se o ambiente virtual está ativo
which python
pip list

# Reinstalar a biblioteca
pip uninstall xxx
pip install xxx
```

### Erro: spaCy modelo não encontrado
```bash
# Baixar modelo manualmente
python -m spacy download pt_core_news_sm

# Verificar modelos instalados
python -m spacy info
```

### Erro: NLTK data não encontrado
```python
import nltk
nltk.download('all')  # Baixa todos os recursos (pode demorar)

# Ou específicos
nltk.download(['punkt', 'stopwords', 'wordnet', 'vader_lexicon'])
```

### Erro: Jupyter não inicia
```bash
# Reinstalar Jupyter
pip uninstall jupyter
pip install jupyter

# Verificar porta
jupyter notebook --port=8889
```

### Problemas de Memória
```python
# Configurar para usar menos memória
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Transformers
```

## 🐳 Docker (Opcional)

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Baixar modelos
RUN python -m spacy download pt_core_news_sm
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

COPY . .
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  nlp-curso:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - .:/app
    environment:
      - JUPYTER_ENABLE_LAB=yes
```

## 📱 Configuração do Jupyter

### Extensões Úteis
```bash
# JupyterLab (interface moderna)
pip install jupyterlab

# Extensões
pip install jupyterlab-git
pip install jupyterlab-variableinspector

# Iniciar JupyterLab
jupyter lab
```

### Configuração Personalizada
```python
# ~/.jupyter/jupyter_notebook_config.py
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
c.NotebookApp.ip = '0.0.0.0'
```

## 🔄 Atualizações

### Manter Bibliotecas Atualizadas
```bash
# Verificar versões desatualizadas
pip list --outdated

# Atualizar todas (cuidado!)
pip freeze | cut -d'=' -f1 | xargs pip install -U

# Atualizar específicas
pip install --upgrade transformers spacy nltk
```

### Backup do Ambiente
```bash
# Exportar ambiente atual
pip freeze > requirements_backup.txt
conda env export > environment_backup.yml
```

## 📞 Suporte

### Se ainda tiver problemas:

1. **Verifique a documentação** de cada biblioteca
2. **Consulte o FAQ** no repositório
3. **Abra uma issue** no GitHub
4. **Procure ajuda** na comunidade

### Links Úteis
- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Jupyter Documentation](https://jupyter.org/documentation)

---

**Próximo passo**: Após a instalação, abra o arquivo `modulo_01_fundamentos/01_fundamentos_nlp.ipynb` para começar o curso! 