# Resumidor Hierarquico com Reinsercao de Conteudo (GPU local)

Pipeline de sumarizacao hierarquica focado em:
- maxima fidelidade factual (numeros, datas, nomes, causalidade);
- controle explicito de proporcao de reducao;
- sintese de partes nao essenciais;
- reinsercao de conteudo critico quando fatos se perdem entre niveis.

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

3. **Reinsercao de conteudo critico**
   - Detecta sentencas criticas por heuristicas (numeros, unidades, citacoes, entidades e termos raros).
   - Verifica cobertura no resumo.
   - Reescreve o resumo reinserindo fatos ausentes sem transformar em lista.

4. **Controle de proporcao final**
   - Ajusta o tamanho final (compressao/expansao controlada) para ficar na faixa:
     `target_reduction_ratio +- reduction_tolerance`.

5. **Execucao em GPU propria**
   - `device_map=auto`
   - `torch_dtype` configuravel (`auto`, `float16`, `bfloat16`, `float32`)
   - suporte a quantizacao 4-bit e 8-bit (BitsAndBytes) quando desejado.

## Instalacao

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Uso (CLI)

Arquivo CLI: `src/cli.py`

```bash
python "src/cli.py" \
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
  --temperature 0.0
```

Saida: JSON com
- `summary`: texto resumido final
- `metrics`: tokens de origem, tokens de resumo, razao atingida, niveis usados etc.

## Parametros recomendados para maxima fidelidade

Para seu objetivo ("maxima eficacia + maxima fidelidade + respeito a proporcao"):

- `temperature=0.0` (determinismo e menor risco de alucinacao)
- `target_reduction_ratio` entre `0.15` e `0.30` (mais baixo = mais informacao)
- `reduction_tolerance` pequeno (`0.02` a `0.04`)
- `critical_sentences_per_chunk` entre `3` e `5`
- `max_reinsertions_per_chunk` entre `2` e `4`
- `overlap_tokens` entre `150` e `260` para textos tecnicos longos
- `max_levels` entre `3` e `5` (conforme tamanho do documento)

## Correcoes e melhorias planejadas (proxima iteracao)

1. **Pontuacao de fidelidade automatica**
   - Adicionar validacao de cobertura factual (numeros/entidades) por nivel.
2. **Reinsercao semantica mais robusta**
   - Trocar heuristica de palavras-chave por matching semantico via embeddings.
3. **Orcamento por secao**
   - Alocar proporcao de resumo por relevancia de secao (nao uniforme).
4. **Suporte a lotes e streaming**
   - Processar documentos grandes em fila mantendo rastreabilidade de metricas.
5. **Suite de testes**
   - Casos com texto juridico, tecnico, financeiro e medico para medir regressao de fidelidade.
