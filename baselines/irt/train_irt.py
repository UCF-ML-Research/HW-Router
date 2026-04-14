import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

try:
    from . import MIRT
except ImportError:
    import MIRT


def normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def build_llm_col_map(columns: List[str], llm_names: List[str]) -> Dict[str, str]:
    llm_lookup = {normalize_name(name): name for name in llm_names}
    col_map = {}
    for col in columns:
        if not col.endswith("_score"):
            continue
        base = col[:-6]
        key = normalize_name(base)
        if key in llm_lookup:
            col_map[col] = llm_lookup[key]
    return col_map


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        all_embeddings.append(pooled.cpu().numpy())
    return np.vstack(all_embeddings)


def load_llm_profiles(llm_profile_path: str) -> Dict[str, str]:
    llm_df = pd.read_csv(llm_profile_path)
    if "name" not in llm_df.columns or "profile" not in llm_df.columns:
        raise ValueError("llm profile file must include columns: name, profile")
    return dict(zip(llm_df["name"], llm_df["profile"]))


def to_long_format(
    df: pd.DataFrame,
    llm_col_map: Dict[str, str],
) -> pd.DataFrame:
    score_cols = list(llm_col_map.keys())
    if not score_cols:
        raise ValueError("no score columns matched the LLM profile names")
    id_vars = [col for col in ["prompt_id", "prompt", "p_tokens"] if col in df.columns]
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=score_cols,
        var_name="llm_col",
        value_name="performance",
    )
    long_df["llm"] = long_df["llm_col"].map(llm_col_map)
    long_df["performance"] = pd.to_numeric(long_df["performance"], errors="coerce")
    long_df = long_df.dropna(subset=["llm", "performance", "prompt"])
    return long_df


def split_indices(n: int, test_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    split = int(n * (1 - test_split))
    return indices[:split], indices[split:]


def train_router(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.data_path)
    llm_profiles = load_llm_profiles(args.llm_profile_path)
    llm_names = list(llm_profiles.keys())

    if args.llm_col_map_path:
        with open(args.llm_col_map_path, "r", encoding="utf-8") as f:
            llm_col_map = json.load(f)
    else:
        llm_col_map = build_llm_col_map(df.columns.tolist(), llm_names)

    long_df = to_long_format(df, llm_col_map)

    used_llms = sorted(long_df["llm"].unique().tolist())
    missing_profiles = [name for name in used_llms if name not in llm_profiles]
    if missing_profiles:
        raise ValueError(f"missing llm profiles for: {missing_profiles}")

    unique_prompts = long_df["prompt"].unique().tolist()
    prompt_embeddings = embed_texts(
        unique_prompts,
        model_name=args.embed_model,
        batch_size=args.embed_batch_size,
        device=args.device,
        max_length=args.max_length,
    )
    prompt_emb_map = dict(zip(unique_prompts, prompt_embeddings))

    llm_texts = [llm_profiles[name] for name in used_llms]
    llm_embeddings = embed_texts(
        llm_texts,
        model_name=args.embed_model,
        batch_size=args.embed_batch_size,
        device=args.device,
        max_length=args.max_length,
    )
    llm_emb_map = dict(zip(used_llms, llm_embeddings))

    llm_vectors = np.stack([llm_emb_map[name] for name in long_df["llm"]])
    prompt_vectors = np.stack([prompt_emb_map[prompt] for prompt in long_df["prompt"]])
    y = long_df["performance"].to_numpy(dtype=np.float32)

    if args.normalize_scores or y.min() < 0 or y.max() > 1:
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    y = np.clip(y, 0.0, 1.0)

    train_idx, test_idx = split_indices(len(y), args.test_split, args.seed)

    train_set = DataLoader(
        TensorDataset(
            torch.tensor(llm_vectors[train_idx], dtype=torch.float32),
            torch.tensor(prompt_vectors[train_idx], dtype=torch.float32),
            torch.tensor(y[train_idx], dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_set = None
    if len(test_idx) > 0:
        test_set = DataLoader(
            TensorDataset(
                torch.tensor(llm_vectors[test_idx], dtype=torch.float32),
                torch.tensor(prompt_vectors[test_idx], dtype=torch.float32),
                torch.tensor(y[test_idx], dtype=torch.float32),
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )

    llm_dim = llm_vectors.shape[1]
    query_dim = prompt_vectors.shape[1]
    cdm = MIRT.MIRT(llm_dim, query_dim, args.latent_dim)
    cdm.train(train_set, test_set, epoch=args.epochs, device=args.device, lr=args.lr)
    cdm.save(args.checkpoint)

    meta = {
        "embed_model": args.embed_model,
        "latent_dim": args.latent_dim,
        "llm_dim": llm_dim,
        "query_dim": query_dim,
        "llm_profile_path": args.llm_profile_path,
        "used_llms": used_llms,
    }
    meta_path = f"{os.path.splitext(args.checkpoint)[0]}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved metadata: {meta_path}")


def predict_router(args: argparse.Namespace) -> None:
    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    llm_profiles = load_llm_profiles(meta["llm_profile_path"])
    used_llms = meta["used_llms"]
    llm_texts = [llm_profiles[name] for name in used_llms]

    llm_embeddings = embed_texts(
        llm_texts,
        model_name=meta["embed_model"],
        batch_size=args.embed_batch_size,
        device=args.device,
        max_length=args.max_length,
    )
    llm_emb_map = dict(zip(used_llms, llm_embeddings))

    prompt_embedding = embed_texts(
        [args.prompt],
        model_name=meta["embed_model"],
        batch_size=1,
        device=args.device,
        max_length=args.max_length,
    )[0]

    cdm = MIRT.MIRT(meta["llm_dim"], meta["query_dim"], meta["latent_dim"])
    state = torch.load(args.checkpoint, map_location=args.device)
    cdm.irt_net.load_state_dict(state)

    predictions = {}
    with torch.no_grad():
        for name in used_llms:
            llm_vec = torch.tensor(llm_emb_map[name], dtype=torch.float32).unsqueeze(0)
            prompt_vec = torch.tensor(prompt_embedding, dtype=torch.float32).unsqueeze(0)
            pred = cdm.generate(llm_vec, prompt_vec, device=args.device)[0]
            predictions[name] = float(pred)

    best_llm = max(predictions.items(), key=lambda x: x[1])
    print(f"best_llm: {best_llm[0]}")
    print(f"predicted_quality: {best_llm[1]:.4f}")
    if args.print_all:
        for name, score in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}\t{score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and predict an IRT router from score columns.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a MIRT router.")
    train_parser.add_argument("--data-path", default="data/UMR_router_training_data.csv")
    train_parser.add_argument("--llm-profile-path", default="baselines/irt/llm_profile.csv")
    train_parser.add_argument("--llm-col-map-path", default=None)
    train_parser.add_argument("--embed-model", default="bert-base-uncased")
    train_parser.add_argument("--embed-batch-size", type=int, default=16)
    train_parser.add_argument("--max-length", type=int, default=512)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--latent-dim", type=int, default=25)
    train_parser.add_argument("--test-split", type=float, default=0.1)
    train_parser.add_argument("--normalize-scores", action="store_true")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--checkpoint", default="baselines/irt/mirt_hw.snapshot")
    train_parser.add_argument("--device", default="auto")

    predict_parser = subparsers.add_parser("predict", help="Predict best LLM for a prompt.")
    predict_parser.add_argument("--prompt", required=True)
    predict_parser.add_argument("--checkpoint", default="baselines/irt/mirt_hw.snapshot")
    predict_parser.add_argument("--meta-path", default="baselines/irt/mirt_hw.meta.json")
    predict_parser.add_argument("--embed-batch-size", type=int, default=16)
    predict_parser.add_argument("--max-length", type=int, default=512)
    predict_parser.add_argument("--print-all", action="store_true")
    predict_parser.add_argument("--device", default="auto")

    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    if args.command == "train":
        train_router(args)
    elif args.command == "predict":
        predict_router(args)


if __name__ == "__main__":
    main()
