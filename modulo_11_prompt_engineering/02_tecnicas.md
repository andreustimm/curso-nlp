[↑ Voltar para o índice](../README.md) | [← Voltar para o módulo](README.md)

# Técnicas Avançadas de Prompting

## 1. Zero-shot Prompting

Zero-shot prompting é a técnica mais básica onde pedimos ao modelo para realizar uma tarefa sem fornecer exemplos.

**Exemplo:**
```
Classifique o seguinte texto como positivo, negativo ou neutro:
"O novo restaurante tem uma decoração incrível, mas os preços são muito altos."
```

## 2. Few-shot Prompting

Few-shot prompting envolve fornecer alguns exemplos antes da tarefa principal.

**Exemplo:**
```
Classifique os seguintes textos como POSITIVO, NEGATIVO ou NEUTRO:

Texto: "O filme foi uma perda de tempo."
Classificação: NEGATIVO

Texto: "O café está na temperatura ideal."
Classificação: POSITIVO

Texto: "A loja abre às 10h."
Classificação: NEUTRO

Agora classifique:
Texto: "A comida estava boa, mas o serviço foi terrível."
```

## 3. Chain-of-Thought (CoT) Prompting

Chain-of-Thought permite que o modelo mostre seu raciocínio passo a passo.

**Exemplo:**
```
Resolva o seguinte problema mostrando seu raciocínio passo a passo:

Em uma loja, há 120 laranjas. No primeiro dia, foram vendidas 1/3 das laranjas. 
No segundo dia, metade das laranjas restantes foi vendida. 
Quantas laranjas sobraram?

Vamos resolver passo a passo:
1. Primeiro, calculamos quantas laranjas foram vendidas no primeiro dia
2. Depois, identificamos quantas sobraram após o primeiro dia
3. Em seguida, calculamos quantas foram vendidas no segundo dia
4. Finalmente, determinamos quantas laranjas restaram
```

## 4. Self-Consistency

Esta técnica envolve gerar múltiplas soluções para o mesmo problema e escolher a mais consistente.

**Exemplo:**
```
Resolva este problema de três maneiras diferentes:

Um trem viaja a 60 km/h. Quanto tempo levará para percorrer 150 km?

Abordagem 1: Usando a fórmula tempo = distância/velocidade
Abordagem 2: Usando proporção
Abordagem 3: Usando regra de três
```

## 5. Prompt Chaining

Prompt Chaining envolve quebrar uma tarefa complexa em subtarefas menores e encadeá-las.

**Exemplo:**
```
Tarefa 1: Extraia as palavras-chave do seguinte texto:
[texto]

Tarefa 2: Para cada palavra-chave extraída, forneça uma definição breve.

Tarefa 3: Use as palavras-chave e definições para criar um resumo estruturado.
```

## 6. ReAct Prompting

ReAct (Reasoning and Acting) combina raciocínio com ações específicas.

**Exemplo:**
```
Objetivo: Encontre a capital da França e liste 3 principais pontos turísticos.

Pensamento 1: Preciso primeiro identificar a capital da França.
Ação 1: A capital da França é Paris.

Pensamento 2: Agora preciso listar pontos turísticos famosos de Paris.
Ação 2: Três principais pontos turísticos são:
1. Torre Eiffel
2. Museu do Louvre
3. Arco do Triunfo

Pensamento 3: Vou verificar se estes são realmente os mais relevantes.
Ação 3: Confirmado, estes são alguns dos pontos turísticos mais visitados de Paris.
```

## 7. Role Prompting

Role Prompting é uma técnica onde definimos um papel específico para o modelo assumir ao responder.

**Exemplo:**
```
Role: Você é um professor de história especializado em Idade Média
Público: Alunos do ensino médio
Tarefa: Explique o sistema feudal de forma envolvente e educativa

Requisitos:
- Use analogias com a sociedade atual
- Inclua fatos históricos interessantes
- Mantenha uma linguagem adequada para adolescentes
- Faça perguntas reflexivas ao longo da explicação
```

## 8. Técnicas de Refinamento

O refinamento de prompts envolve a melhoria iterativa das instruções para obter resultados mais precisos.

**Exemplo:**
```
# Versão 1 (Básica)
Resuma este artigo científico.

# Versão 2 (Melhorada)
Faça um resumo deste artigo científico em 3 parágrafos:
- Primeiro parágrafo: Objetivo e metodologia
- Segundo parágrafo: Principais resultados
- Terceiro parágrafo: Conclusões e implicações

# Versão 3 (Refinada)
Analise este artigo científico seguindo estas diretrizes:

Formato:
1. Resumo executivo (2-3 frases)
2. Metodologia
   - Abordagem utilizada
   - Tamanho da amostra
   - Período do estudo
3. Resultados principais
   - Descobertas chave
   - Dados estatísticos relevantes
4. Conclusões
   - Implicações práticas
   - Sugestões para pesquisas futuras

Restrições:
- Use linguagem técnica apropriada
- Mantenha objetividade
- Destaque limitações do estudo
- Inclua números e percentuais relevantes
```

## Dicas para Escolher a Técnica Adequada

- **Zero-shot**: Use para tarefas simples e diretas
- **Few-shot**: Ideal quando você tem exemplos claros e quer consistência
- **Chain-of-Thought**: Melhor para problemas complexos que exigem raciocínio
- **Self-Consistency**: Útil quando a precisão é crucial
- **Prompt Chaining**: Ótimo para decompor tarefas complexas
- **ReAct**: Excelente para tarefas que combinam raciocínio e ação

## Exercícios

1. Crie um prompt usando Chain-of-Thought para resolver um problema de lógica
2. Desenvolva um exemplo de Few-shot prompting para classificação de emails
3. Implemente um ReAct prompt para criar um plano de estudos
4. Use Self-Consistency para resolver um problema matemático de três maneiras diferentes

[↑ Voltar para o índice](../README.md) | [← Voltar para o módulo](README.md)

[← Anterior: Fundamentos](01_fundamentos.md) | [Próximo: Casos de Uso e Aplicações →](03_aplicacoes.md)