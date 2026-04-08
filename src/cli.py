from __future__ import annotations

import argparse
from pathlib import Path

from hierarchical_summarizer import HierarchicalSummarizer, SummarizationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumidor hierarquico com reinsercao de conteudo critico."
    )
    parser.add_argument("--input-file", required=True, help="Caminho do arquivo de entrada.")
    parser.add_argument(
        "--output-file",
        default="",
        help="Caminho para salvar o resultado em JSON. Se vazio, imprime no stdout.",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=1400)
    parser.add_argument("--overlap-tokens", type=int, default=180)
    parser.add_argument("--max-levels", type=int, default=4)
    parser.add_argument("--merge-group-size", type=int, default=4)
    parser.add_argument("--target-reduction-ratio", type=float, default=0.20)
    parser.add_argument("--reduction-tolerance", type=float, default=0.03)
    parser.add_argument("--min-local-ratio", type=float, default=0.45)
    parser.add_argument("--max-local-ratio", type=float, default=0.90)
    parser.add_argument("--critical-sentences-per-chunk", type=int, default=3)
    parser.add_argument("--max-reinsertions-per-chunk", type=int, default=2)
    parser.add_argument("--max-generation-tokens", type=int, default=420)
    parser.add_argument("--length-adjustment-rounds", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {input_path}")

    config = SummarizationConfig(
        model_name=args.model_name,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        max_input_tokens=args.max_input_tokens,
        overlap_tokens=args.overlap_tokens,
        max_levels=args.max_levels,
        merge_group_size=args.merge_group_size,
        target_reduction_ratio=args.target_reduction_ratio,
        reduction_tolerance=args.reduction_tolerance,
        min_local_ratio=args.min_local_ratio,
        max_local_ratio=args.max_local_ratio,
        critical_sentences_per_chunk=args.critical_sentences_per_chunk,
        max_reinsertions_per_chunk=args.max_reinsertions_per_chunk,
        max_generation_tokens=args.max_generation_tokens,
        length_adjustment_rounds=args.length_adjustment_rounds,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    summarizer = HierarchicalSummarizer(config)
    result = summarizer.summarize(text)
    payload = summarizer.to_pretty_json(result)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.write_text(payload + "\n", encoding="utf-8")
        print(f"Resumo salvo em: {output_path}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
