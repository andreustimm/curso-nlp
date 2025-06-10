"""
Utilitários comuns para NLP - Curso de NLP
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import unicodedata

def limpar_texto_basico(texto):
    """
    Limpeza básica de texto
    """
    if not isinstance(texto, str):
        return ""
    
    # Converter para minúsculas
    texto = texto.lower()
    
    # Remover caracteres especiais, manter apenas letras e espaços
    texto = re.sub(r'[^a-záàâãéêíóôõúç\s]', '', texto)
    
    # Remover espaços extras
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

def remover_acentos(texto):
    """
    Remove acentos de caracteres
    """
    if not isinstance(texto, str):
        return ""
    
    # Normalizar unicode e remover acentos
    texto_normalizado = unicodedata.normalize('NFD', texto)
    texto_sem_acentos = ''.join(char for char in texto_normalizado 
                               if unicodedata.category(char) != 'Mn')
    
    return texto_sem_acentos

def extrair_urls(texto):
    """
    Extrai URLs de um texto
    """
    padrao_url = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(padrao_url, texto)
    return urls

def extrair_emails(texto):
    """
    Extrai emails de um texto
    """
    padrao_email = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(padrao_email, texto)
    return emails

def extrair_mencoes(texto):
    """
    Extrai menções (@usuario) de um texto
    """
    padrao_mencao = r'@\w+'
    mencoes = re.findall(padrao_mencao, texto)
    return mencoes

def extrair_hashtags(texto):
    """
    Extrai hashtags (#tag) de um texto
    """
    padrao_hashtag = r'#\w+'
    hashtags = re.findall(padrao_hashtag, texto)
    return hashtags

def contar_palavras(textos):
    """
    Conta frequência de palavras em uma lista de textos
    """
    todas_palavras = []
    for texto in textos:
        if isinstance(texto, str):
            palavras = texto.split()
            todas_palavras.extend(palavras)
    
    return Counter(todas_palavras)

def gerar_nuvem_palavras(textos, titulo="Nuvem de Palavras", figsize=(12, 8)):
    """
    Gera uma nuvem de palavras a partir de uma lista de textos
    """
    # Juntar todos os textos
    texto_completo = ' '.join([str(texto) for texto in textos if isinstance(texto, str)])
    
    if not texto_completo.strip():
        print("Nenhum texto válido para gerar nuvem de palavras")
        return
    
    # Criar nuvem de palavras
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(texto_completo)
    
    # Plotar
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(titulo, fontsize=16)
    plt.tight_layout()
    plt.show()

def analisar_comprimento_textos(textos):
    """
    Analisa comprimento de uma lista de textos
    """
    comprimentos = []
    for texto in textos:
        if isinstance(texto, str):
            comprimentos.append(len(texto))
        else:
            comprimentos.append(0)
    
    df_analise = pd.DataFrame({
        'texto': textos,
        'comprimento': comprimentos,
        'num_palavras': [len(str(texto).split()) for texto in textos]
    })
    
    return df_analise

def plotar_distribuicao_comprimentos(textos, titulo="Distribuição de Comprimentos"):
    """
    Plota distribuição de comprimentos de textos
    """
    df_analise = analisar_comprimento_textos(textos)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histograma de caracteres
    axes[0].hist(df_analise['comprimento'], bins=10, alpha=0.7, color='skyblue')
    axes[0].set_title('Distribuição: Número de Caracteres')
    axes[0].set_xlabel('Caracteres')
    axes[0].set_ylabel('Frequência')
    
    # Histograma de palavras
    axes[1].hist(df_analise['num_palavras'], bins=10, alpha=0.7, color='lightgreen')
    axes[1].set_title('Distribuição: Número de Palavras')
    axes[1].set_xlabel('Palavras')
    axes[1].set_ylabel('Frequência')
    
    plt.suptitle(titulo)
    plt.tight_layout()
    plt.show()
    
    return df_analise

def comparar_antes_depois(textos_antes, textos_depois, labels=['Antes', 'Depois']):
    """
    Compara estatísticas antes e depois do pré-processamento
    """
    stats_antes = analisar_comprimento_textos(textos_antes)
    stats_depois = analisar_comprimento_textos(textos_depois)
    
    # Criar DataFrame comparativo
    comparacao = pd.DataFrame({
        f'{labels[0]}_caracteres': stats_antes['comprimento'],
        f'{labels[1]}_caracteres': stats_depois['comprimento'],
        f'{labels[0]}_palavras': stats_antes['num_palavras'],
        f'{labels[1]}_palavras': stats_depois['num_palavras']
    })
    
    print("Estatísticas comparativas:")
    print(comparacao.describe())
    
    # Plotar comparação
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Caracteres - antes vs depois
    axes[0,0].hist(stats_antes['comprimento'], alpha=0.7, label=labels[0], color='blue')
    axes[0,0].hist(stats_depois['comprimento'], alpha=0.7, label=labels[1], color='red')
    axes[0,0].set_title('Distribuição: Caracteres')
    axes[0,0].legend()
    
    # Palavras - antes vs depois
    axes[0,1].hist(stats_antes['num_palavras'], alpha=0.7, label=labels[0], color='blue')
    axes[0,1].hist(stats_depois['num_palavras'], alpha=0.7, label=labels[1], color='red')
    axes[0,1].set_title('Distribuição: Palavras')
    axes[0,1].legend()
    
    # Box plot comparativo - caracteres
    data_chars = [stats_antes['comprimento'], stats_depois['comprimento']]
    axes[1,0].boxplot(data_chars, labels=labels)
    axes[1,0].set_title('Box Plot: Caracteres')
    
    # Box plot comparativo - palavras
    data_words = [stats_antes['num_palavras'], stats_depois['num_palavras']]
    axes[1,1].boxplot(data_words, labels=labels)
    axes[1,1].set_title('Box Plot: Palavras')
    
    plt.tight_layout()
    plt.show()
    
    return comparacao

def detectar_idioma_simples(texto):
    """
    Detecção simples de idioma baseada em palavras comuns
    """
    # Palavras comuns por idioma
    palavras_pt = ['o', 'a', 'de', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no']
    palavras_en = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with']
    palavras_es = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da']
    
    texto_lower = texto.lower()
    palavras = texto_lower.split()
    
    score_pt = sum(1 for palavra in palavras if palavra in palavras_pt)
    score_en = sum(1 for palavra in palavras if palavra in palavras_en)
    score_es = sum(1 for palavra in palavras if palavra in palavras_es)
    
    scores = {'português': score_pt, 'inglês': score_en, 'espanhol': score_es}
    
    if max(scores.values()) == 0:
        return 'desconhecido'
    
    return max(scores, key=scores.get)

def calcular_estatisticas_corpus(textos):
    """
    Calcula estatísticas gerais de um corpus
    """
    total_textos = len(textos)
    total_caracteres = sum(len(str(texto)) for texto in textos)
    total_palavras = sum(len(str(texto).split()) for texto in textos)
    
    # Vocabulário único
    todas_palavras = []
    for texto in textos:
        palavras = str(texto).split()
        todas_palavras.extend(palavras)
    
    vocabulario_unico = len(set(todas_palavras))
    palavra_mais_comum = Counter(todas_palavras).most_common(1)
    
    stats = {
        'total_textos': total_textos,
        'total_caracteres': total_caracteres,
        'total_palavras': total_palavras,
        'vocabulario_unico': vocabulario_unico,
        'avg_caracteres_por_texto': total_caracteres / total_textos if total_textos > 0 else 0,
        'avg_palavras_por_texto': total_palavras / total_textos if total_textos > 0 else 0,
        'palavra_mais_comum': palavra_mais_comum[0] if palavra_mais_comum else None
    }
    
    return stats

def print_estatisticas_corpus(textos, titulo="Estatísticas do Corpus"):
    """
    Imprime estatísticas do corpus de forma formatada
    """
    stats = calcular_estatisticas_corpus(textos)
    
    print(f"\n{titulo}")
    print("=" * 50)
    print(f"Total de textos: {stats['total_textos']:,}")
    print(f"Total de caracteres: {stats['total_caracteres']:,}")
    print(f"Total de palavras: {stats['total_palavras']:,}")
    print(f"Vocabulário único: {stats['vocabulario_unico']:,}")
    print(f"Média de caracteres por texto: {stats['avg_caracteres_por_texto']:.1f}")
    print(f"Média de palavras por texto: {stats['avg_palavras_por_texto']:.1f}")
    
    if stats['palavra_mais_comum']:
        palavra, freq = stats['palavra_mais_comum']
        print(f"Palavra mais comum: '{palavra}' ({freq:,} vezes)")
    
    print("=" * 50)

# Função para adicionar caminho dos datasets
def adicionar_caminho_datasets():
    """
    Adiciona o caminho dos datasets ao Python path
    """
    import sys
    import os
    
    # Caminho para a pasta datasets
    datasets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')
    
    if datasets_path not in sys.path:
        sys.path.append(datasets_path)
    
    return datasets_path 