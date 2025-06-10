"""
Módulo de utilitários para o Curso de NLP
"""

from .nlp_utils import (
    limpar_texto_basico,
    remover_acentos,
    extrair_urls,
    extrair_emails,
    extrair_mencoes,
    extrair_hashtags,
    contar_palavras,
    gerar_nuvem_palavras,
    analisar_comprimento_textos,
    plotar_distribuicao_comprimentos,
    comparar_antes_depois,
    detectar_idioma_simples,
    calcular_estatisticas_corpus,
    print_estatisticas_corpus,
    adicionar_caminho_datasets
)

__all__ = [
    'limpar_texto_basico',
    'remover_acentos',
    'extrair_urls',
    'extrair_emails',
    'extrair_mencoes',
    'extrair_hashtags',
    'contar_palavras',
    'gerar_nuvem_palavras',
    'analisar_comprimento_textos',
    'plotar_distribuicao_comprimentos',
    'comparar_antes_depois',
    'detectar_idioma_simples',
    'calcular_estatisticas_corpus',
    'print_estatisticas_corpus',
    'adicionar_caminho_datasets'
] 