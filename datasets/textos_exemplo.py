"""
Datasets de exemplo para o curso de NLP
"""

# Textos simples para demonstrações básicas
textos_basicos = [
    "Inteligência artificial está transformando o mundo dos negócios.",
    "Machine learning permite que computadores aprendam sem programação explícita.",
    "Deep learning usa redes neurais profundas para resolver problemas complexos.",
    "Natural Language Processing ajuda computadores a entender linguagem humana.",
    "O futuro da tecnologia é muito promissor e cheio de oportunidades."
]

# Textos com ruído para limpeza
textos_com_ruido = [
    "Olá!!! Este é um texto COM MUITOS problemas... 😀😃",
    "Visite nosso site em https://exemplo.com para mais informações!!!",
    "@usuario mencionou #NLP no tweet: 'Adorei o curso!' 👍",
    "DESCONTO DE 50%!!! Ligue agora: (11) 99999-9999",
    "E-mail: contato@empresa.com.br - Resposta em 24h!!!",
    "<p>HTML tags não devem aparecer no texto final</p>",
    "Texto    com     espaços        irregulares",
    "Números misturados: 123abc456def789 e datas: 01/01/2023"
]

# Reviews de produtos (análise de sentimentos)
reviews_produtos = [
    "Produto excelente! Superou minhas expectativas. Recomendo muito!",
    "Qualidade péssima, chegou quebrado. Não comprem!",
    "Produto ok, nada demais. Preço justo para o que oferece.",
    "Adorei! Produto de ótima qualidade, entrega rápida.",
    "Horrível! Dinheiro jogado fora. Atendimento péssimo.",
    "Bom custo-benefício. Produto funcional, sem grandes problemas.",
    "Produto maravilhoso! Melhor compra que já fiz online.",
    "Mais ou menos... Funciona, mas poderia ser melhor.",
    "Decepcionante. Esperava muito mais pela descrição.",
    "Perfeito! Exatamente como descrito. Vendedor confiável."
]

# Textos jornalísticos
noticias = [
    "O governo anunciou hoje novas medidas econômicas para combater a inflação. As medidas incluem redução de impostos e incentivos ao setor produtivo.",
    "A tecnologia de inteligência artificial está sendo aplicada na medicina para diagnósticos mais precisos. Hospitais já reportam melhorias significativas.",
    "O mercado financeiro reagiu positivamente às últimas decisões do Banco Central. O dólar recuou e as ações subiram na bolsa de valores.",
    "Cientistas descobriram uma nova espécie de planta na Amazônia. A descoberta pode contribuir para o desenvolvimento de novos medicamentos.",
    "A educação digital ganhou impulso durante a pandemia. Universidades investem em plataformas online para melhorar o ensino à distância."
]

# Textos acadêmicos
textos_academicos = [
    "A metodologia proposta neste estudo utiliza técnicas de aprendizado de máquina para análise de grandes volumes de dados textuais.",
    "Os resultados obtidos demonstram que a abordagem baseada em redes neurais profundas supera os métodos tradicionais em 15%.",
    "Este trabalho apresenta uma revisão sistemática da literatura sobre processamento de linguagem natural aplicado à área médica.",
    "A análise estatística dos dados coletados revela correlações significativas entre as variáveis estudadas (p < 0.05).",
    "Pesquisas futuras devem considerar a implementação de modelos mais robustos para lidar com a variabilidade dos dados reais."
]

# Conversas de chat/suporte
conversas_chat = [
    "Oi! Preciso de ajuda com meu pedido",
    "Claro! Qual é o número do seu pedido?",
    "É o pedido #12345. Não chegou ainda",
    "Deixe-me verificar... Vejo que está em transporte",
    "Ok, quando vai chegar?",
    "Deve chegar até amanhã às 18h",
    "Perfeito! Obrigado pela ajuda :)",
    "De nada! Algo mais que posso ajudar?",
    "Não, só isso mesmo. Tenha um bom dia!",
    "Você também! Até mais!"
]

# Textos em diferentes idiomas (para detecção de idioma)
textos_multilinguagem = [
    "Este é um texto em português brasileiro.",
    "This is a text written in English language.",
    "Esto es un texto escrito en español.",
    "Ceci est un texte écrit en français.",
    "Dies ist ein Text in deutscher Sprache.",
    "Questo è un testo scritto in italiano.",
    "Este é um texto em português de Portugal."
]

# Códigos e dados técnicos (para limpeza específica)
textos_tecnicos = [
    "Erro 404: Página não encontrada. Status code: HTTP_404_NOT_FOUND",
    "SELECT * FROM usuarios WHERE id = 123 AND status = 'ativo'",
    "function processText(input) { return input.toLowerCase().trim(); }",
    "pip install numpy==1.21.0 pandas>=1.3.0 scikit-learn",
    "Log: [2023-12-01 10:30:15] INFO - Processo executado com sucesso",
    "URL: https://api.exemplo.com/v1/usuarios?page=1&limit=10",
    "JSON: {\"nome\": \"João\", \"idade\": 30, \"ativo\": true}"
]

def get_dataset(nome):
    """
    Retorna um dataset específico pelo nome
    
    Datasets disponíveis:
    - basicos: Textos simples para demonstrações
    - com_ruido: Textos que precisam de limpeza
    - reviews: Avaliações de produtos
    - noticias: Textos jornalísticos
    - academicos: Textos acadêmicos
    - chat: Conversas de suporte
    - multilinguagem: Textos em vários idiomas
    - tecnicos: Códigos e dados técnicos
    """
    datasets = {
        'basicos': textos_basicos,
        'com_ruido': textos_com_ruido,
        'reviews': reviews_produtos,
        'noticias': noticias,
        'academicos': textos_academicos,
        'chat': conversas_chat,
        'multilinguagem': textos_multilinguagem,
        'tecnicos': textos_tecnicos
    }
    
    if nome in datasets:
        return datasets[nome]
    else:
        print(f"Dataset '{nome}' não encontrado.")
        print(f"Datasets disponíveis: {list(datasets.keys())}")
        return []

def listar_datasets():
    """Lista todos os datasets disponíveis"""
    datasets = [
        'basicos', 'com_ruido', 'reviews', 'noticias',
        'academicos', 'chat', 'multilinguagem', 'tecnicos'
    ]
    
    print("Datasets disponíveis:")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")
    
    return datasets 