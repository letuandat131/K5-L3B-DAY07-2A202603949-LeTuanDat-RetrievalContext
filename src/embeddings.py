"""Pinned local ONNX MiniLM embedder (384 dimensions) with deterministic caching."""
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .ingestion import download_if_missing


class LocalMiniLMEmbedder:
    """Offline-capable ONNX MiniLM embedder matching Day 06 space (384d, cosine)."""

    def __init__(self, root: Path) -> None:
        config = json.loads((root / "configs/experiment.json").read_text(encoding="utf-8"))["model"]
        self.model_name = config["name"]
        self.dimensions = int(config["dimensions"])
        self.config = config

        cache_dir = root / ".cache" / ("model-" + config["revision"][:8])
        cache_dir.mkdir(parents=True, exist_ok=True)
        day6_cache = Path(r"C:\Users\letua\Desktop\cứ xóa\K5-L3B-DAY06-2A202603949-DataIngestion\.cache") / (
            "model-" + config["revision"][:8]
        )

        for rel_name, checksum in config["files"].items():
            target = cache_dir / Path(rel_name).name
            if not target.exists() and (day6_cache / Path(rel_name).name).exists():
                src_file = day6_cache / Path(rel_name).name
                if hashlib.sha256(src_file.read_bytes()).hexdigest() == checksum:
                    shutil.copy2(src_file, target)
            url = f"https://huggingface.co/{self.model_name}/resolve/{config['revision']}/{rel_name}"
            download_if_missing(url, target, checksum)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(config["threads"])
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(cache_dir / "model_quantized.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(str(cache_dir / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.cache_path = cache_dir / "embeddings.json"
        self.fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
        saved = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}
        if saved and saved.get("fingerprint") != self.fingerprint:
            raise ValueError("Embedding cache fingerprint mismatch")
        self.cache: dict[str, list[float]] = saved.get("vectors", {})
        self.usage = {"computed_texts": 0, "cache_hits": 0, "inference_seconds": 0.0}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Embedding inputs must be non-empty strings")
        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self.cache]
        unique_missing: dict[str, str] = {}
        for k, t in missing:
            unique_missing.setdefault(k, t)
        self.usage["cache_hits"] += len(texts) - len(missing)

        items = list(unique_missing.items())
        bs = int(self.config["batch_size"])
        self.tokenizer.enable_truncation(max_length=int(self.config["max_tokens"]))
        for start in range(0, len(items), bs):
            batch = items[start : start + bs]
            encoded = self.tokenizer.encode_batch([t for _, t in batch])
            feed = {
                "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64),
            }
            tick = time.perf_counter()
            out = self.session.run(None, {i.name: feed[i.name] for i in self.session.get_inputs()})[0]
            mask = feed["attention_mask"][..., None]
            pooled = (out * mask).sum(axis=1) / mask.sum(axis=1)
            vecs = (pooled / np.linalg.norm(pooled, axis=1, keepdims=True)).astype(np.float32)
            self.usage["inference_seconds"] += time.perf_counter() - tick
            self.usage["computed_texts"] += len(batch)
            for (k, _), vec in zip(batch, vecs):
                self.cache[k] = vec.tolist()
        if unique_missing:
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"fingerprint": self.fingerprint, "vectors": self.cache}), encoding="utf-8")
            tmp.replace(self.cache_path)
        return [self.cache[k] for k in keys]

    def embed_query(self, question: str) -> list[float]:
        return self.embed_documents([question])[0]
