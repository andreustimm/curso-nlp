[↑ Voltar para o índice](../README.md) | [← Voltar para o módulo](README.md)

# Casos de Uso e Aplicações Práticas

## 1. Classificação de Texto

### Análise de Sentimento
```python
# Prompt usando Few-shot com exemplos balanceados
prompt = """
Classifique o sentimento dos seguintes textos como POSITIVO, NEGATIVO ou NEUTRO:

Exemplo 1: "Este produto quebrou no primeiro uso."
Sentimento: NEGATIVO

Exemplo 2: "A entrega foi feita no prazo previsto."
Sentimento: NEUTRO

Exemplo 3: "Adorei o atendimento, muito profissional!"
Sentimento: POSITIVO

Agora classifique: "{texto_do_usuario}"
"""
```

### Categorização de Documentos
```python
# Prompt usando Zero-shot com estrutura clara
prompt = """
Categorize o seguinte documento em UMA das categorias:
- Financeiro
- Recursos Humanos
- Marketing
- Tecnologia
- Legal

Documento:
{documento}

Forneça a categoria e uma breve justificativa.
"""
```

## 2. Geração de Código

### Criação de Funções
```python
# Prompt usando Chain-of-Thought
prompt = """
Crie uma função Python que {descrição_da_função}.

Siga estes passos:
1. Defina os parâmetros necessários
2. Documente a função usando docstrings
3. Implemente a lógica principal
4. Adicione tratamento de erros
5. Forneça um exemplo de uso

Mostre seu raciocínio em cada etapa.
"""

# Exemplo de uso:
"""
Crie uma função Python que converta uma lista de temperaturas de Celsius para Fahrenheit.

1. Parâmetros:
   - Precisamos de uma lista de números (temperaturas em Celsius)
   - Opcional: um parâmetro para arredondamento

2. Docstring:
   '''
   Converte uma lista de temperaturas de Celsius para Fahrenheit.
   
   Args:
       temps_c (list): Lista de temperaturas em Celsius
       round_digits (int, optional): Número de casas decimais. Default: 2
   
   Returns:
       list: Lista de temperaturas em Fahrenheit
   '''

3. Lógica:
   - Fórmula: F = C * 9/5 + 32
   - Aplicar a fórmula em cada elemento da lista
   - Arredondar conforme especificado

4. Tratamento de erros:
   - Verificar se a entrada é uma lista
   - Verificar se todos os elementos são números
   - Validar o parâmetro de arredondamento

5. Implementação:
"""
def celsius_to_fahrenheit(temps_c, round_digits=2):
    '''
    Converte uma lista de temperaturas de Celsius para Fahrenheit.
    
    Args:
        temps_c (list): Lista de temperaturas em Celsius
        round_digits (int, optional): Número de casas decimais. Default: 2
    
    Returns:
        list: Lista de temperaturas em Fahrenheit
    '''
    if not isinstance(temps_c, list):
        raise TypeError("A entrada deve ser uma lista")
    
    if not all(isinstance(x, (int, float)) for x in temps_c):
        raise ValueError("Todos os elementos devem ser números")
    
    if not isinstance(round_digits, int) or round_digits < 0:
        raise ValueError("round_digits deve ser um inteiro não-negativo")
    
    return [round(temp * 9/5 + 32, round_digits) for temp in temps_c]
```

### Debugging
```python
# Prompt usando ReAct
prompt = """
Analise e corrija o seguinte código Python:

{código_com_erro}

Pensamento 1: Identificar possíveis erros no código
Ação 1: Listar todos os problemas encontrados

Pensamento 2: Propor correções
Ação 2: Mostrar o código corrigido

Pensamento 3: Explicar as correções
Ação 3: Detalhar por que cada correção foi necessária

Pensamento 4: Sugerir melhorias
Ação 4: Propor otimizações ou boas práticas adicionais
"""
```

## 3. Análise de Sentimento

### Análise Detalhada
```python
# Prompt usando Prompt Chaining
prompts = [
    # Etapa 1: Identificação de Aspectos
    """
    Identifique os principais aspectos mencionados nesta avaliação de produto:
    {avaliação}
    
    Liste cada aspecto em uma nova linha.
    """,
    
    # Etapa 2: Análise de Sentimento por Aspecto
    """
    Para cada aspecto identificado, determine o sentimento (Positivo/Negativo/Neutro)
    e a intensidade (1-5):
    
    Aspectos:
    {aspectos}
    
    Formato da resposta:
    Aspecto: [Sentimento] (Intensidade) - Justificativa
    """,
    
    # Etapa 3: Resumo Geral
    """
    Com base na análise anterior, crie um resumo estruturado:
    
    1. Pontos Positivos:
    2. Pontos Negativos:
    3. Sentimento Geral:
    4. Recomendações:
    """
]
```

## 4. Extração de Informações

### Extração de Dados Estruturados
```python
# Prompt usando Few-shot com formato específico
prompt = """
Extraia as informações relevantes do texto e formate como JSON:

Exemplo:
Texto: "Entre em contato com João Silva pelo email joao@email.com ou telefone (11) 98765-4321"
Saída:
{
    "nome": "João Silva",
    "contatos": {
        "email": "joao@email.com",
        "telefone": "(11) 98765-4321"
    }
}

Agora extraia do seguinte texto:
{texto_do_usuario}
"""
```

## 5. Resolução de Problemas Matemáticos

### Problemas Complexos
```python
# Prompt usando Chain-of-Thought com Self-Consistency
prompt = """
Resolva o seguinte problema matemático de três maneiras diferentes:

{problema}

Método 1: Álgebra Direta
- Passo a passo da solução algébrica

Método 2: Visualização
- Resolução usando diagramas ou representação visual
- Explicação de cada etapa

Método 3: Decomposição
- Quebrar o problema em partes menores
- Resolver cada parte separadamente
- Combinar as soluções

Compare os resultados e verifique a consistência.
"""
```

## Exercícios Práticos

1. **Classificação de Emails**
   - Crie um sistema de classificação de emails em categorias (Urgente, Importante, Normal, Spam)
   - Use Few-shot prompting com exemplos diversos
   - Implemente tratamento para casos ambíguos

2. **Geração de Código Documentado**
   - Desenvolva prompts para gerar código bem documentado
   - Inclua testes unitários
   - Adicione comentários explicativos

3. **Análise de Feedback de Clientes**
   - Crie um sistema para analisar feedback de clientes
   - Extraia temas principais
   - Identifique sentimentos por aspecto
   - Gere recomendações acionáveis

4. **Resolução de Problemas de Lógica**
   - Use Chain-of-Thought para resolver problemas de lógica
   - Implemente verificação de consistência
   - Forneça explicações detalhadas

## Melhores Práticas

1. **Validação de Resultados**
   - Sempre verifique a consistência das respostas
   - Implemente verificações de qualidade
   - Use múltiplos métodos quando possível

2. **Tratamento de Erros**
   - Prepare prompts para lidar com entradas inesperadas
   - Forneça feedback claro sobre erros
   - Implemente fallbacks quando necessário

3. **Otimização de Prompts**
   - Teste diferentes variações
   - Meça a qualidade das respostas
   - Ajuste com base nos resultados

4. **Documentação**
   - Mantenha um registro de prompts efetivos
   - Documente casos de uso bem-sucedidos
   - Compartilhe aprendizados com a equipe

## Próximos Passos

No próximo capítulo, abordaremos boas práticas e técnicas de otimização para melhorar ainda mais seus prompts.

[↑ Voltar para o índice](../README.md) | [← Voltar para o módulo](README.md)

[← Anterior: Técnicas de Prompting](02_tecnicas.md) | [Próximo: Boas Práticas e Otimização →](04_boas_praticas.md) 