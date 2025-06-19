[↑ Voltar para o índice](../README.md) | [← Voltar para o módulo](README.md)

# Fundamentos do Prompt Engineering

## O que é Prompt Engineering?

Prompt Engineering é a arte e ciência de projetar e otimizar instruções (prompts) para modelos de linguagem large (LLMs) com o objetivo de obter os melhores resultados possíveis para uma determinada tarefa.

## Elementos Básicos de um Prompt

### 1. Instrução
A parte do prompt que especifica o que você quer que o modelo faça.

**Exemplo:**
```
Traduza o seguinte texto para o português:
"The weather is beautiful today."
```

### 2. Contexto
Informações adicionais que ajudam o modelo a entender melhor a tarefa.

**Exemplo:**
```
Contexto: Você é um especialista em medicina cardiovascular escrevendo para pacientes leigos.
Tarefa: Explique o que é pressão arterial alta de forma simples e clara.
```

### 3. Exemplos (quando necessário)
Demonstrações do tipo de resposta esperada.

**Exemplo:**
```
Converta estas temperaturas de Celsius para Fahrenheit:

Exemplo:
Entrada: 0°C
Saída: 32°F

Agora converta:
25°C
```

### 4. Formato de Saída
Especificação de como você quer que a resposta seja formatada.

**Exemplo:**
```
Analise o sentimento do seguinte texto e responda no formato JSON:
"Este produto superou todas as minhas expectativas!"

Formato desejado:
{
    "sentimento": "positivo/negativo/neutro",
    "confiança": 0-1,
    "palavras_chave": []
}
```

## Configurações de LLM

### Temperature
- **0.0**: Mais determinístico, melhor para tarefas que exigem consistência
- **0.7**: Bom equilíbrio entre criatividade e coerência
- **1.0**: Mais criativo, melhor para brainstorming e geração de ideias

### Max Tokens
- Limite o número de tokens para controlar o tamanho da resposta
- Importante para manter respostas concisas e relevantes

## Dicas Gerais para Design de Prompts

### 1. Seja Específico e Claro
❌ Ruim:
```
Fale sobre gatos.
```

✅ Bom:
```
Descreva 3 características únicas dos gatos domésticos que os diferenciam de outros felinos, focando em seus comportamentos sociais.
```

### 2. Use Delimitadores
```
Resumo o seguinte texto entre triplas aspas:
"""
Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
"""
```

### 3. Estruture a Informação
```
Analise o seguinte e-mail:
TÍTULO: Reunião de projeto
DE: joao@empresa.com
PARA: maria@empresa.com
CORPO: Podemos remarcar nossa reunião para amanhã às 15h?

Por favor, identifique:
1. Assunto principal
2. Urgência (alta/média/baixa)
3. Ação necessária
```

### 4. Especifique o Papel
```
Atue como um professor de matemática do ensino médio explicando o teorema de Pitágoras para um aluno que está tendo dificuldades com geometria.
```

### 5. Use Step-by-Step
```
Resolva este problema de programação passo a passo:
1. Primeiro, entenda o problema
2. Liste as entradas e saídas esperadas
3. Desenvolva a lógica
4. Escreva o código
5. Teste com exemplos
```

## Problemas Comuns e Soluções

### 1. Respostas Muito Longas
**Solução:** Especifique um limite
```
Explique o que é fotossíntese em no máximo 3 frases.
```

### 2. Respostas Vagas
**Solução:** Peça exemplos específicos
```
Explique o conceito de 'machine learning' e forneça 2 exemplos concretos de aplicações no mundo real.
```

### 3. Falta de Estrutura
**Solução:** Forneça um template
```
Analise este filme seguindo a estrutura:
- Título:
- Gênero:
- Pontos fortes:
- Pontos fracos:
- Nota (0-10):
```

## Exercícios Práticos

1. Crie um prompt para gerar um resumo de um artigo científico
2. Desenvolva um prompt para análise de sentimento
3. Escreva um prompt para geração de código Python
4. Crie um prompt para explicar um conceito complexo de forma simples

## Próximos Passos

No próximo capítulo, exploraremos técnicas avançadas de prompting que nos permitirão realizar tarefas mais complexas e obter resultados ainda melhores.

[↑ Voltar para o índice](../README.md) | [← Voltar para o módulo](README.md)

[Próximo: Técnicas de Prompting →](02_tecnicas.md) 