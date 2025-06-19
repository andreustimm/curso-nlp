[↑ Voltar para o índice](../README.md) | [← Voltar para o módulo](README.md)

# Exercícios Práticos de Prompt Engineering

## Exercício 1: Análise de Sentimento

### Objetivo
Criar um sistema de análise de sentimento usando diferentes técnicas de prompting.

### Tarefas
1. **Zero-shot Prompting**
```python
# Crie um prompt que analise o sentimento sem exemplos
prompt_1 = """
Analise o sentimento do seguinte texto e classifique como POSITIVO, NEGATIVO ou NEUTRO:

"{texto}"

Explique sua classificação em um parágrafo.
"""

# Teste com:
textos = [
    "O novo sistema é incrível e superou todas as expectativas!",
    "O produto chegou com defeito e o suporte não ajudou.",
    "A entrega foi realizada no prazo previsto."
]
```

2. **Few-shot Prompting**
```python
# Adicione exemplos para melhorar a precisão
prompt_2 = """
Classifique o sentimento dos textos como POSITIVO, NEGATIVO ou NEUTRO:

Exemplo 1: "Adorei o produto, recomendo!"
Classificação: POSITIVO
Razão: Expressa satisfação e recomendação.

Exemplo 2: "Não gostei do atendimento."
Classificação: NEGATIVO
Razão: Indica insatisfação com o serviço.

Exemplo 3: "O restaurante abre às 19h."
Classificação: NEUTRO
Razão: Apenas informa um fato, sem opinião.

Agora classifique: "{texto}"
"""
```

3. **Análise Detalhada**
```python
# Crie um prompt para análise mais profunda
prompt_3 = """
Faça uma análise detalhada do seguinte texto:

"{texto}"

Forneça:
1. Sentimento geral (POSITIVO/NEGATIVO/NEUTRO)
2. Intensidade (1-5)
3. Aspectos específicos mencionados
4. Palavras-chave que influenciaram a análise
5. Sugestões baseadas no feedback

Formato da resposta:
{
    "sentimento": "",
    "intensidade": 0,
    "aspectos": [],
    "palavras_chave": [],
    "sugestoes": []
}
"""
```

## Exercício 2: Geração de Código

### Objetivo
Desenvolver prompts para geração de código com diferentes níveis de complexidade.

### Tarefas
1. **Função Simples**
```python
# Prompt para função básica
prompt_1 = """
Crie uma função Python que {descrição_da_função}.

Requisitos:
1. Use docstring para documentação
2. Inclua tratamento de erros
3. Adicione exemplos de uso

Exemplo de descrição: "calcule a média de uma lista de números"
"""
```

2. **Classe Completa**
```python
# Prompt para classe com métodos
prompt_2 = """
Crie uma classe Python para {descrição_da_classe} com os seguintes requisitos:

1. Atributos:
   - Liste os atributos necessários
   - Defina tipos apropriados
   - Inclua valores padrão quando apropriado

2. Métodos:
   - Construtor
   - Getters/Setters necessários
   - Métodos específicos da classe
   - Método __str__ para representação

3. Documentação:
   - Docstring da classe
   - Docstring dos métodos
   - Exemplos de uso

4. Tratamento de Erros:
   - Validações apropriadas
   - Exceções personalizadas se necessário

Exemplo de descrição: "gerenciar uma lista de tarefas (TODO list)"
"""
```

3. **Projeto Estruturado**
```python
# Prompt para estrutura de projeto
prompt_3 = """
Crie a estrutura básica para um projeto Python que {descrição_do_projeto}.

Inclua:
1. Estrutura de diretórios
2. Arquivos principais
3. Classes e funções necessárias
4. Testes unitários
5. Requirements.txt
6. README.md

Exemplo de descrição: "API REST para um sistema de blog"
"""
```

## Exercício 3: Análise de Texto

### Objetivo
Criar prompts para diferentes tipos de análise textual.

### Tarefas
1. **Resumo de Texto**
```python
prompt_1 = """
Crie um resumo do seguinte texto em três níveis de detalhe:

Texto:
"{texto_longo}"

Forneça:
1. Resumo em uma frase (máximo 20 palavras)
2. Resumo em um parágrafo (máximo 50 palavras)
3. Resumo detalhado (máximo 150 palavras)

Para cada resumo, mantenha:
- Pontos principais
- Contexto essencial
- Conclusões importantes
"""
```

2. **Extração de Informações**
```python
prompt_2 = """
Extraia informações estruturadas do seguinte texto:

"{texto}"

Formate a saída como JSON com:
1. Entidades mencionadas (pessoas, organizações, lugares)
2. Datas e horários
3. Números e valores
4. Conceitos-chave
5. Relacionamentos entre entidades

Exemplo de formato:
{
    "entidades": {
        "pessoas": [],
        "organizacoes": [],
        "lugares": []
    },
    "temporal": {
        "datas": [],
        "horarios": []
    },
    "numericos": {
        "valores": [],
        "quantidades": []
    },
    "conceitos": [],
    "relacionamentos": []
}
"""
```

3. **Análise Comparativa**
```python
prompt_3 = """
Compare os seguintes textos e analise suas similaridades e diferenças:

Texto 1:
"{texto_1}"

Texto 2:
"{texto_2}"

Forneça:
1. Temas comuns
2. Diferenças principais
3. Análise de estilo
4. Conclusões gerais

Use esta estrutura:
1. Similaridades:
   - Tema
   - Argumentos
   - Conclusões

2. Diferenças:
   - Abordagem
   - Perspectiva
   - Detalhes específicos

3. Análise Estilística:
   - Tom
   - Vocabulário
   - Estrutura

4. Conclusão:
   - Síntese
   - Observações importantes
"""
```

## Exercício 4: Projeto Final

### Objetivo
Desenvolver um sistema completo de análise de feedback de clientes.

### Tarefas
1. **Coleta e Preparação**
```python
prompt_1 = """
Prepare os dados do seguinte feedback para análise:

Feedback: "{texto_feedback}"

1. Limpeza:
   - Remova informações irrelevantes
   - Corrija erros óbvios
   - Normalize formato

2. Estruturação:
   - Identifique componentes principais
   - Separe em seções lógicas
   - Marque pontos-chave

3. Categorização:
   - Tipo de feedback
   - Urgência
   - Área relacionada
"""
```

2. **Análise Detalhada**
```python
prompt_2 = """
Realize uma análise completa do feedback preparado:

1. Sentimento:
   - Geral
   - Por aspecto
   - Tendências

2. Problemas:
   - Identificação
   - Priorização
   - Impacto

3. Sugestões:
   - Do cliente
   - Implícitas
   - Potenciais soluções

4. Oportunidades:
   - Melhorias
   - Inovações
   - Vantagens competitivas
"""
```

3. **Geração de Relatório**
```python
prompt_3 = """
Crie um relatório executivo baseado na análise:

1. Sumário Executivo:
   - Principais descobertas
   - Métricas importantes
   - Recomendações prioritárias

2. Análise Detalhada:
   - Dados quantitativos
   - Insights qualitativos
   - Tendências identificadas

3. Recomendações:
   - Ações imediatas
   - Médio prazo
   - Longo prazo

4. Próximos Passos:
   - Plano de ação
   - Responsabilidades
   - Prazos

Formato: Relatório profissional em Markdown
"""
```

## Avaliação

### Critérios
- Qualidade dos prompts (30%)
- Efetividade dos resultados (30%)
- Criatividade e inovação (20%)
- Documentação e organização (20%)

### Entrega
1. Código fonte dos prompts
2. Exemplos de resultados
3. Documentação de uso
4. Análise de performance
5. Sugestões de melhorias

## Recursos Adicionais

### Ferramentas Recomendadas
- Jupyter Notebook para testes
- GitHub para versionamento
- Ferramentas de análise de texto
- Bibliotecas Python relevantes

### Material de Referência
- Documentação de LLMs
- Artigos sobre Prompt Engineering
- Estudos de caso
- Melhores práticas da indústria

[↑ Voltar para o índice](../README.md) | [← Voltar para o módulo](README.md)

[← Anterior: Boas Práticas e Otimização](04_boas_praticas.md) 