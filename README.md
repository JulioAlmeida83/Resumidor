# Resumidor Hierarquico com Reinsercao de Conteudo (GPU local)

Pipeline de sumarizacao hierarquica focado em:
- maxima fidelidade factual (numeros, datas, nomes, causalidade);
- controle explicito de proporcao de reducao;
- sintese de partes nao essenciais;
- reinsercao de conteudo critico quando fatos se perdem entre niveis.
- avaliacao automatica de fidelidade factual (numeros, entidades e cobertura de sentencas criticas).

## Estado atual (diagnostico)

Este repositorio estava praticamente vazio. Foi implementada uma base funcional em Python para rodar em GPU propria usando Hugging Face Transformers.

## Arquitetura implementada

Arquivo principal: `src/hierarchical_summarizer.py`

1. **Chunking por sentencas + overlap de contexto**
   - Divide texto em blocos por limite de tokens (`max_input_tokens`).
   - Mantem sobreposicao (`overlap_tokens`) para reduzir perda de continuidade.

2. **Sumarizacao hierarquica multi-nivel**
   - Resume cada chunk, agrupa resultados, repete por ate `max_levels`.
   - O fator de compressao local e calculado para aproximar a reducao global alvo.

3. **Reinsercao de conteudo critico (lexical + semantica)**
   - Detecta sentencas criticas por heuristicas (numeros, unidades, citacoes, entidades e termos raros).
   - Verifica cobertura no resumo com:
     - matching lexical (palavras-chave, entidades, numeros);
     - similaridade semantica por embeddings (`sentence-transformers`).
   - Reescreve o resumo reinserindo fatos ausentes sem transformar em lista.

4. **Avaliacao automatica de fidelidade**
   - Gera score composto no JSON de saida:
     - `overall_faithfulness_score`
     - `numeric_coverage`
     - `entity_coverage`
     - `critical_semantic_coverage`
     - `mean_critical_similarity`

5. **Controle de proporcao final**
   - Ajusta o tamanho final (compressao/expansao controlada) para ficar na faixa:
     `target_reduction_ratio +- reduction_tolerance`.

6. **Execucao em GPU propria**
   - `device_map=auto`
   - `torch_dtype` configuravel (`auto`, `float16`, `bfloat16`, `float32`)
   - suporte a quantizacao 4-bit e 8-bit (BitsAndBytes) quando desejado.

## Instalacao

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Uso (CLI)

Arquivo CLI: `src/cli.py`

```bash
python3 "src/cli.py" \
  --input-file "examples/input.txt" \
  --output-file "examples/output.json" \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --device-map auto \
  --torch-dtype bfloat16 \
  --target-reduction-ratio 0.20 \
  --reduction-tolerance 0.03 \
  --max-levels 4 \
  --merge-group-size 4 \
  --max-input-tokens 1400 \
  --overlap-tokens 180 \
  --critical-sentences-per-chunk 3 \
  --max-reinsertions-per-chunk 2 \
  --semantic-model-name "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
  --semantic-similarity-threshold 0.56 \
  --factual-eval-top-sentences 24 \
  --temperature 0.0
```

Saida: JSON com
- `summary`: texto resumido final
- `metrics`: tokens de origem, tokens de resumo, razao atingida, niveis usados etc.
- `metrics.factual_fidelity`: score de fidelidade automatica com submetricas.

## Parametros recomendados para maxima fidelidade

Para seu objetivo ("maxima eficacia + maxima fidelidade + respeito a proporcao"):

- `temperature=0.0` (determinismo e menor risco de alucinacao)
- `target_reduction_ratio` entre `0.15` e `0.30` (mais baixo = mais informacao)
- `reduction_tolerance` pequeno (`0.02` a `0.04`)
- `critical_sentences_per_chunk` entre `3` e `5`
- `max_reinsertions_per_chunk` entre `2` e `4`
- `semantic_similarity_threshold` entre `0.52` e `0.62`
- `overlap_tokens` entre `150` e `260` para textos tecnicos longos
- `max_levels` entre `3` e `5` (conforme tamanho do documento)

## Flags novas da CLI

- `--disable-semantic-reinsertion`  
  Desliga matching semantico e usa apenas estrategia lexical.
- `--semantic-model-name`  
  Modelo de embeddings para cobertura semantica.
- `--semantic-similarity-threshold`  
  Limiar de cobertura semantica para considerar uma sentenca critica como preservada.
- `--semantic-batch-size`  
  Batch da etapa de embeddings.
- `--factual-eval-top-sentences`  
  Quantidade de sentencas criticas usadas na pontuacao de fidelidade.
