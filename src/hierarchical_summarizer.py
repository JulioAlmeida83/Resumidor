from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


PT_STOPWORDS = {
    "a",
    "as",
    "ao",
    "aos",
    "aquela",
    "aquele",
    "de",
    "da",
    "das",
    "do",
    "dos",
    "e",
    "em",
    "entre",
    "era",
    "essa",
    "esse",
    "esta",
    "este",
    "foi",
    "ha",
    "isso",
    "isto",
    "ja",
    "la",
    "mais",
    "mas",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "por",
    "que",
    "se",
    "sem",
    "ser",
    "sua",
    "suas",
    "tambem",
    "tem",
    "tendo",
    "ter",
    "teu",
    "teus",
    "um",
    "uma",
}


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


@dataclass
class SummarizationConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    max_input_tokens: int = 1400
    overlap_tokens: int = 180
    max_levels: int = 4
    merge_group_size: int = 4
    target_reduction_ratio: float = 0.20
    reduction_tolerance: float = 0.03
    min_local_ratio: float = 0.45
    max_local_ratio: float = 0.90
    critical_sentences_per_chunk: int = 3
    max_reinsertions_per_chunk: int = 2
    max_generation_tokens: int = 420
    length_adjustment_rounds: int = 2
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    language: str = "pt-BR"
    semantic_reinsertion_enabled: bool = True
    semantic_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    semantic_similarity_threshold: float = 0.56
    semantic_batch_size: int = 32
    factual_eval_top_sentences: int = 24

    def validate(self) -> None:
        if not 0 < self.target_reduction_ratio < 1:
            raise ValueError("target_reduction_ratio must be between 0 and 1.")
        if not 0 <= self.reduction_tolerance < self.target_reduction_ratio:
            raise ValueError("reduction_tolerance must be >= 0 and < target_reduction_ratio.")
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Use either 4-bit or 8-bit quantization, not both.")
        if self.merge_group_size < 2:
            raise ValueError("merge_group_size must be >= 2.")
        if self.max_levels < 1:
            raise ValueError("max_levels must be >= 1.")
        if not 0.1 <= self.semantic_similarity_threshold <= 0.95:
            raise ValueError("semantic_similarity_threshold must be between 0.1 and 0.95.")
        if self.semantic_batch_size < 1:
            raise ValueError("semantic_batch_size must be >= 1.")
        if self.factual_eval_top_sentences < 1:
            raise ValueError("factual_eval_top_sentences must be >= 1.")


class HierarchicalSummarizer:
    def __init__(self, config: SummarizationConfig) -> None:
        self.config = config
        self.config.validate()

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_config = AutoConfig.from_pretrained(self.config.model_name)
        dtype = self._resolve_dtype(self.config.torch_dtype)

        model_kwargs = {"device_map": self.config.device_map, "torch_dtype": dtype}
        if self.config.load_in_4bit or self.config.load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
            )

        if model_config.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name, **model_kwargs)
            self.is_encoder_decoder = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **model_kwargs)
            self.is_encoder_decoder = False
        self.model.eval()
        self.generation_device = self._resolve_generation_device()

        self.semantic_model = None
        if self.config.semantic_reinsertion_enabled:
            self._load_semantic_model()

    @staticmethod
    def _resolve_dtype(dtype_name: str):
        dtype_name = dtype_name.lower()
        if dtype_name == "auto":
            return "auto"
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float32":
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    @staticmethod
    def _best_compute_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_generation_device(self) -> torch.device:
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for mapped_device in device_map.values():
                if isinstance(mapped_device, str):
                    if mapped_device.startswith("cuda"):
                        return torch.device(mapped_device)
                    if mapped_device == "mps":
                        return torch.device("mps")
                    if mapped_device == "cpu":
                        return torch.device("cpu")
        model_device = getattr(self.model, "device", None)
        if isinstance(model_device, torch.device) and model_device.type != "meta":
            return model_device
        return torch.device(self._best_compute_device())

    def _load_semantic_model(self) -> None:
        if SentenceTransformer is None:
            warnings.warn(
                "sentence-transformers is not installed. Falling back to lexical reinsertion coverage.",
                RuntimeWarning,
            )
            return
        device = self._best_compute_device()
        try:
            self.semantic_model = SentenceTransformer(self.config.semantic_model_name, device=device)
        except Exception as exc:
            warnings.warn(
                f"Failed to load semantic model '{self.config.semantic_model_name}'. "
                f"Using lexical fallback instead. Error: {exc}",
                RuntimeWarning,
            )
            self.semantic_model = None

    def token_len(self, text: str) -> int:
        if not text.strip():
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[\.\!\?\u2026])\s+", text.strip())
        return [s.strip() for s in sentences if s and s.strip()]

    def _split_long_sentence(self, sentence: str) -> List[str]:
        max_tokens = self.config.max_input_tokens
        if self.token_len(sentence) <= max_tokens:
            return [sentence]
        words = sentence.split()
        parts: List[str] = []
        current: List[str] = []
        for word in words:
            tentative = " ".join(current + [word])
            if current and self.token_len(tentative) > max_tokens:
                parts.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            parts.append(" ".join(current))
        return parts if parts else [sentence]

    def chunk_text(self, text: str) -> List[str]:
        sentences: List[str] = []
        for sentence in self._split_sentences(text):
            sentences.extend(self._split_long_sentence(sentence))

        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0
        max_tokens = self.config.max_input_tokens

        for sentence in sentences:
            sent_tokens = self.token_len(sentence)
            if current and current_tokens + sent_tokens > max_tokens:
                chunks.append(" ".join(current))
                overlap_sentences: List[str] = []
                overlap_token_count = 0
                for prev_sentence in reversed(current):
                    prev_tokens = self.token_len(prev_sentence)
                    if overlap_token_count + prev_tokens > self.config.overlap_tokens:
                        break
                    overlap_sentences.insert(0, prev_sentence)
                    overlap_token_count += prev_tokens
                current = overlap_sentences + [sentence]
                current_tokens = overlap_token_count + sent_tokens
            else:
                current.append(sentence)
                current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _word_tokens(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9À-ÿ][A-Za-z0-9À-ÿ\-_/]*", text.lower())

    def _extract_numbers(self, text: str) -> Set[str]:
        numbers = re.findall(
            r"\b\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?\b|\b\d+(?:[.,]\d+)?%?\b|R\$\s?\d+(?:[.,]\d+)?",
            text,
            flags=re.IGNORECASE,
        )
        normalized: Set[str] = set()
        for num in numbers:
            normalized_num = re.sub(r"\s+", "", num.lower())
            normalized.add(normalized_num)
        return normalized

    def _extract_entities(self, text: str, max_items: int = 60) -> Set[str]:
        entity_pattern = r"\b[A-ZÀ-Ý]{2,}\b|\b[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+"
        entities = re.findall(entity_pattern, text)
        cleaned = [re.sub(r"\s+", " ", ent.strip()) for ent in entities if ent and len(ent) > 2]
        if not cleaned:
            return set()
        counts = Counter(cleaned)
        return {entity for entity, _ in counts.most_common(max_items)}

    def _critical_sentences(self, text: str, top_k: int) -> List[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        if len(sentences) <= top_k:
            return sentences

        all_words = [w for s in sentences for w in self._word_tokens(s) if w not in PT_STOPWORDS and len(w) > 2]
        global_counts = Counter(all_words)

        scored: List[Tuple[float, str]] = []
        for sentence in sentences:
            score = 0.0
            if re.search(r"\d", sentence):
                score += 2.0
            if re.search(r"\b(?:R\$|\$|%|km|kg|h|min|seg|ms|GHz|GB|MB|TB|kWh)\b", sentence, flags=re.IGNORECASE):
                score += 1.2
            if re.search(r"['\"“”‘’]", sentence):
                score += 0.7
            if re.search(r"\b[A-Z]{2,}\b", sentence):
                score += 0.8
            if re.search(r"\b[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+", sentence):
                score += 0.8

            rare_bonus = 0.0
            words = [w for w in self._word_tokens(sentence) if w not in PT_STOPWORDS and len(w) > 2]
            for word in words:
                if global_counts[word] <= 2:
                    rare_bonus += 0.25
            score += min(2.0, rare_bonus)
            score += min(1.2, len(words) / 18.0)
            scored.append((score, sentence))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [sentence for _, sentence in scored[:top_k]]

    def _keywords(self, text: str, max_terms: int = 6) -> List[str]:
        words = [w for w in self._word_tokens(text) if w not in PT_STOPWORDS and len(w) > 3]
        freq = Counter(words)
        return [word for word, _ in freq.most_common(max_terms)]

    def _missing_critical_sentences_lexical(self, summary: str, critical_sentences: List[str]) -> List[str]:
        summary_lower = summary.lower()
        missing: List[str] = []
        summary_numbers = self._extract_numbers(summary)
        for sentence in critical_sentences:
            sentence_numbers = self._extract_numbers(sentence)
            if sentence_numbers and not sentence_numbers.issubset(summary_numbers):
                missing.append(sentence)
                continue
            keys = self._keywords(sentence, max_terms=4)
            if not keys:
                continue
            hits = sum(1 for key in keys if key in summary_lower)
            if hits == 0:
                missing.append(sentence)
        return missing

    def _encode_semantic(self, texts: List[str]) -> torch.Tensor:
        if self.semantic_model is None:
            return torch.empty((0, 0), dtype=torch.float32)
        return self.semantic_model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.config.semantic_batch_size,
        )

    def _missing_critical_sentences_semantic(self, summary: str, critical_sentences: List[str]) -> List[str]:
        if self.semantic_model is None:
            return self._missing_critical_sentences_lexical(summary, critical_sentences)
        summary_sentences = self._split_sentences(summary)
        if not summary_sentences:
            return critical_sentences

        summary_lower = summary.lower()
        summary_numbers = self._extract_numbers(summary)
        summary_entities_lower = {entity.lower() for entity in self._extract_entities(summary, max_items=120)}

        summary_embeddings = self._encode_semantic(summary_sentences)
        critical_embeddings = self._encode_semantic(critical_sentences)
        if summary_embeddings.numel() == 0 or critical_embeddings.numel() == 0:
            return self._missing_critical_sentences_lexical(summary, critical_sentences)

        sims = torch.matmul(critical_embeddings, summary_embeddings.T)
        max_sims = sims.max(dim=1).values.tolist()

        missing: List[str] = []
        for idx, sentence in enumerate(critical_sentences):
            sentence_numbers = self._extract_numbers(sentence)
            if sentence_numbers and not sentence_numbers.issubset(summary_numbers):
                missing.append(sentence)
                continue

            sentence_entities = {entity.lower() for entity in self._extract_entities(sentence, max_items=16)}
            entity_hits = sum(1 for entity in sentence_entities if entity in summary_entities_lower)
            lexical_hits = sum(1 for key in self._keywords(sentence, max_terms=4) if key in summary_lower)
            semantic_hit = max_sims[idx] >= self.config.semantic_similarity_threshold

            if not semantic_hit and lexical_hits == 0 and entity_hits == 0:
                missing.append(sentence)
        return missing

    def _missing_critical_sentences(self, summary: str, critical_sentences: List[str]) -> List[str]:
        if not self.config.semantic_reinsertion_enabled:
            return self._missing_critical_sentences_lexical(summary, critical_sentences)
        return self._missing_critical_sentences_semantic(summary, critical_sentences)

    def _format_prompt(self, system_text: str, user_text: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"{system_text}\n\n{user_text}\n\nResposta:"

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {name: tensor.to(self.generation_device) for name, tensor in inputs.items()}

        do_sample = self.config.temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.config.temperature

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        if self.is_encoder_decoder:
            generated_ids = output_ids[0]
        else:
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _summarize_once(self, source_text: str, target_tokens: int) -> str:
        system_prompt = (
            "Voce e um resumidor tecnico altamente fiel. "
            "Nao invente fatos e nao altere numeros, datas, nomes proprios ou causalidade."
        )
        user_prompt = (
            "Resuma o texto com maxima fidelidade factual e boa legibilidade. "
            "Priorize preservar informacoes essenciais e sintetizar partes nao essenciais. "
            f"Tamanho alvo aproximado: {target_tokens} tokens.\n\n"
            f"TEXTO:\n<<<\n{source_text}\n>>>\n\nRESUMO FIEL:"
        )
        prompt = self._format_prompt(system_prompt, user_prompt)
        generated = self._generate(
            prompt,
            max_new_tokens=min(self.config.max_generation_tokens, int(target_tokens * 1.7) + 40),
        )
        return generated if generated else source_text

    def _reinsert_critical(self, source_text: str, summary: str, target_tokens: int) -> str:
        critical = self._critical_sentences(source_text, self.config.critical_sentences_per_chunk)
        missing = self._missing_critical_sentences(summary, critical)
        if not missing:
            return summary
        missing = missing[: self.config.max_reinsertions_per_chunk]

        bullets = "\n".join(f"- {sentence}" for sentence in missing)
        system_prompt = (
            "Voce revisa resumos para preservar fatos criticos sem inventar informacoes. "
            "Mantenha coerencia, concisao e fidelidade."
        )
        user_prompt = (
            f"Resumo atual (alvo ~{target_tokens} tokens):\n{summary}\n\n"
            "Fatos criticos ausentes que devem ser reinseridos, se ainda nao estiverem no texto:\n"
            f"{bullets}\n\n"
            "Reescreva um unico resumo integrado, sem lista separada, mantendo estilo objetivo:"
        )
        prompt = self._format_prompt(system_prompt, user_prompt)
        revised = self._generate(
            prompt,
            max_new_tokens=min(self.config.max_generation_tokens, int(target_tokens * 1.4) + 32),
        )
        return revised or summary

    def _enforce_final_ratio(self, source_text: str, summary: str) -> str:
        source_tokens = max(1, self.token_len(source_text))
        target = max(60, int(source_tokens * self.config.target_reduction_ratio))
        lower = max(40, int(source_tokens * (self.config.target_reduction_ratio - self.config.reduction_tolerance)))
        upper = max(lower + 10, int(source_tokens * (self.config.target_reduction_ratio + self.config.reduction_tolerance)))

        current = summary
        for _ in range(self.config.length_adjustment_rounds):
            current_tokens = self.token_len(current)
            if lower <= current_tokens <= upper:
                return current

            if current_tokens > upper:
                system_prompt = "Voce comprime textos sem perder fatos essenciais."
                user_prompt = (
                    f"Comprima o resumo para ~{target} tokens (faixa {lower}-{upper}). "
                    "Nao remova numeros, datas, nomes e conclusoes centrais.\n\n"
                    f"Resumo atual:\n{current}\n\nResumo comprimido:"
                )
            else:
                system_prompt = "Voce aumenta levemente a cobertura com fidelidade."
                user_prompt = (
                    f"Expanda o resumo para ~{target} tokens (faixa {lower}-{upper}), "
                    "adicionando apenas detalhes relevantes presentes no texto fonte.\n\n"
                    f"Texto fonte:\n{source_text}\n\nResumo atual:\n{current}\n\nResumo revisado:"
                )

            prompt = self._format_prompt(system_prompt, user_prompt)
            updated = self._generate(
                prompt,
                max_new_tokens=min(self.config.max_generation_tokens, int(target * 1.4) + 32),
            )
            if not updated:
                break
            current = updated

        current_tokens = self.token_len(current)
        if current_tokens > upper:
            sentences = self._split_sentences(current)
            compact: List[str] = []
            for sentence in sentences:
                compact.append(sentence)
                if self.token_len(" ".join(compact)) > upper:
                    compact.pop()
                    break
            if compact:
                current = " ".join(compact)
        elif current_tokens < lower:
            missing = self._missing_critical_sentences(current, self._critical_sentences(source_text, 12))
            for sentence in missing:
                tentative = f"{current} {sentence}".strip()
                if self.token_len(tentative) > upper:
                    break
                current = tentative
                if self.token_len(current) >= lower:
                    break
        return current

    def evaluate_factual_fidelity(self, source_text: str, summary_text: str) -> Dict[str, object]:
        source_numbers = self._extract_numbers(source_text)
        summary_numbers = self._extract_numbers(summary_text)
        numbers_present = len([num for num in source_numbers if num in summary_numbers])
        numeric_coverage = _safe_div(numbers_present, len(source_numbers), default=1.0)

        source_entities = self._extract_entities(source_text, max_items=80)
        summary_lower = summary_text.lower()
        entities_present = len([ent for ent in source_entities if ent.lower() in summary_lower])
        entity_coverage = _safe_div(entities_present, len(source_entities), default=1.0)

        critical = self._critical_sentences(source_text, self.config.factual_eval_top_sentences)
        if critical:
            missing = self._missing_critical_sentences(summary_text, critical)
            critical_semantic_coverage = 1.0 - _safe_div(len(missing), len(critical))
        else:
            critical_semantic_coverage = 1.0

        mean_similarity = 0.0
        if self.semantic_model is not None and critical:
            summary_sentences = self._split_sentences(summary_text)
            if summary_sentences:
                critical_embeddings = self._encode_semantic(critical)
                summary_embeddings = self._encode_semantic(summary_sentences)
                if critical_embeddings.numel() > 0 and summary_embeddings.numel() > 0:
                    sims = torch.matmul(critical_embeddings, summary_embeddings.T)
                    mean_similarity = float(sims.max(dim=1).values.mean().item())

        overall_score = (
            0.42 * numeric_coverage
            + 0.23 * entity_coverage
            + 0.35 * critical_semantic_coverage
        )
        overall_score = float(max(0.0, min(1.0, overall_score)))

        return {
            "overall_faithfulness_score": round(overall_score, 4),
            "numeric_coverage": round(numeric_coverage, 4),
            "entity_coverage": round(entity_coverage, 4),
            "critical_semantic_coverage": round(critical_semantic_coverage, 4),
            "mean_critical_similarity": round(mean_similarity, 4),
            "source_facts": {
                "numbers": len(source_numbers),
                "entities": len(source_entities),
                "critical_sentences": len(critical),
            },
        }

    def summarize(self, text: str) -> Dict[str, object]:
        if not text or not text.strip():
            raise ValueError("Input text is empty.")

        chunks = self.chunk_text(text)
        if not chunks:
            raise ValueError("Failed to build chunks from input text.")

        original_tokens = self.token_len(text)
        levels_estimate = max(
            1,
            min(
                self.config.max_levels,
                math.ceil(math.log(max(1, len(chunks)), self.config.merge_group_size)) + 1,
            ),
        )
        local_ratio = self.config.target_reduction_ratio ** (1.0 / levels_estimate)
        local_ratio = min(self.config.max_local_ratio, max(self.config.min_local_ratio, local_ratio))

        current_blocks = chunks[:]
        levels_used = 0

        for _ in range(self.config.max_levels):
            next_summaries: List[str] = []
            for block in current_blocks:
                source_tokens = max(1, self.token_len(block))
                block_target = max(48, int(source_tokens * local_ratio))
                summary = self._summarize_once(block, target_tokens=block_target)
                summary = self._reinsert_critical(block, summary, target_tokens=block_target)
                next_summaries.append(summary)

            levels_used += 1
            if len(next_summaries) == 1:
                current_blocks = next_summaries
                break

            grouped: List[str] = []
            for index in range(0, len(next_summaries), self.config.merge_group_size):
                group_text = "\n\n".join(next_summaries[index : index + self.config.merge_group_size])
                grouped.append(group_text)
            current_blocks = grouped
            if len(current_blocks) == 1:
                break

        candidate = current_blocks[0]
        candidate = self._reinsert_critical(
            text,
            candidate,
            target_tokens=max(60, int(original_tokens * self.config.target_reduction_ratio)),
        )
        final_summary = self._enforce_final_ratio(text, candidate)
        summary_tokens = self.token_len(final_summary)
        factual_metrics = self.evaluate_factual_fidelity(text, final_summary)

        return {
            "summary": final_summary,
            "metrics": {
                "original_tokens": original_tokens,
                "summary_tokens": summary_tokens,
                "achieved_reduction_ratio": round(summary_tokens / max(1, original_tokens), 4),
                "target_reduction_ratio": self.config.target_reduction_ratio,
                "reduction_tolerance": self.config.reduction_tolerance,
                "levels_used": levels_used,
                "initial_chunks": len(chunks),
                "semantic_reinsertion_enabled": bool(self.semantic_model is not None and self.config.semantic_reinsertion_enabled),
                "semantic_model_name": self.config.semantic_model_name if self.semantic_model is not None else "",
                "factual_fidelity": factual_metrics,
            },
        }

    @staticmethod
    def to_pretty_json(data: Dict[str, object]) -> str:
        return json.dumps(data, ensure_ascii=False, indent=2)
