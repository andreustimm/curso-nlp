#!/usr/bin/env python3
"""
Script de configuração inicial para o Curso de NLP
Execute este script para configurar o ambiente e baixar recursos necessários
"""

import subprocess
import sys
import os

def instalar_dependencias():
    """Instala as dependências do requirements.txt"""
    print("📦 Instalando dependências...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependências instaladas com sucesso!")
    except subprocess.CalledProcessError:
        print("❌ Erro ao instalar dependências")
        return False
    return True

def baixar_recursos_nltk():
    """Baixa recursos necessários do NLTK"""
    print("📚 Baixando recursos do NLTK...")
    try:
        import nltk
        
        recursos = [
            'punkt',
            'stopwords', 
            'wordnet',
            'vader_lexicon',
            'rslp',
            'floresta',
            'mac_morpho'
        ]
        
        for recurso in recursos:
            try:
                nltk.download(recurso, quiet=True)
                print(f"  ✅ {recurso}")
            except:
                print(f"  ⚠️ {recurso} (opcional)")
        
        print("✅ Recursos do NLTK baixados!")
    except ImportError:
        print("❌ NLTK não encontrado")
        return False
    return True

def baixar_modelo_spacy():
    """Baixa modelo em português do spaCy"""
    print("🌐 Baixando modelo em português do spaCy...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "pt_core_news_sm"])
        print("✅ Modelo em português do spaCy baixado!")
    except subprocess.CalledProcessError:
        print("⚠️ Erro ao baixar modelo do spaCy. Tentando modelo em inglês...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            print("✅ Modelo em inglês do spaCy baixado!")
        except subprocess.CalledProcessError:
            print("❌ Erro ao baixar modelos do spaCy")
            return False
    return True

def verificar_jupyter():
    """Verifica se Jupyter está instalado"""
    print("📓 Verificando Jupyter...")
    try:
        subprocess.check_call([sys.executable, "-m", "jupyter", "--version"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Jupyter está instalado!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Jupyter não encontrado")
        return False

def criar_estrutura_diretorios():
    """Cria estrutura de diretórios se não existir"""
    print("📁 Verificando estrutura de diretórios...")
    
    diretorios = [
        'modulo_01_fundamentos',
        'modulo_02_preprocessamento', 
        'modulo_03_analise_estatistica',
        'modulo_04_representacao',
        'modulo_05_classificacao',
        'modulo_06_modelos_sequencia',
        'modulo_07_deep_learning',
        'modulo_08_transformers',
        'modulo_09_tarefas_avancadas',
        'modulo_10_projetos',
        'datasets',
        'utils'
    ]
    
    for diretorio in diretorios:
        if not os.path.exists(diretorio):
            os.makedirs(diretorio)
            print(f"  📁 Criado: {diretorio}")
        else:
            print(f"  ✅ Existe: {diretorio}")
    
    print("✅ Estrutura de diretórios verificada!")

def testar_importacoes():
    """Testa se as principais bibliotecas podem ser importadas"""
    print("🧪 Testando importações...")
    
    bibliotecas = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('nltk', None),
        ('spacy', None),
        ('textblob', 'TextBlob'),
        ('sklearn', None)
    ]
    
    sucesso = True
    for biblioteca, alias in bibliotecas:
        try:
            if alias:
                exec(f"import {biblioteca} as {alias}")
            else:
                exec(f"import {biblioteca}")
            print(f"  ✅ {biblioteca}")
        except ImportError:
            print(f"  ❌ {biblioteca}")
            sucesso = False
    
    return sucesso

def main():
    """Função principal de configuração"""
    print("🚀 Configurando Curso de NLP")
    print("=" * 50)
    
    # Verificar Python
    print(f"🐍 Python {sys.version}")
    
    # Criar estrutura
    criar_estrutura_diretorios()
    
    # Instalar dependências
    if not instalar_dependencias():
        print("❌ Falha na instalação de dependências")
        return
    
    # Baixar recursos
    baixar_recursos_nltk()
    baixar_modelo_spacy()
    
    # Verificar Jupyter
    verificar_jupyter()
    
    # Testar importações
    if testar_importacoes():
        print("\n🎉 Configuração concluída com sucesso!")
        print("\n📚 Para começar o curso:")
        print("1. Execute: jupyter notebook")
        print("2. Abra: modulo_01_fundamentos/01_fundamentos_nlp.ipynb")
        print("3. Siga a sequência dos módulos")
    else:
        print("\n⚠️ Algumas bibliotecas não foram instaladas corretamente")
        print("Verifique os erros acima e tente novamente")

if __name__ == "__main__":
    main() 