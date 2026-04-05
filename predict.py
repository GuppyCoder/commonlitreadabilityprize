from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.config import OUTPUT_PATH
from src.datasets import CommonLitDataset
from src.models import CommonLitModel


def get_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def find_latest_timestamp(model_name: str) -> str:
    candidates = []
    for ts_dir in OUTPUT_PATH.iterdir():
        if not ts_dir.is_dir():
            continue
        ckpts = list(ts_dir.glob(f"{model_name}/fold_*/*.ckpt"))
        if ckpts:
            candidates.append(ts_dir.name)

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found under {OUTPUT_PATH} for model '{model_name}'."
        )

    return sorted(candidates)[-1]


def load_checkpoint_bundle(ckpt_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path.parent))
    config = AutoConfig.from_pretrained(str(ckpt_path.parent))
    model = CommonLitModel.load_from_checkpoint(str(ckpt_path), hf_config=config)
    return model, tokenizer


def _collect_ckpt_paths(
    timestamp: str, model_name: str, fold_index: int | None = None
) -> list[Path]:
    root = OUTPUT_PATH / timestamp / model_name
    if fold_index is None:
        ckpt_paths = sorted(root.glob("fold_*/*.ckpt"))
    else:
        ckpt_paths = sorted((root / f"fold_{fold_index}").glob("*.ckpt"))

    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoints found for timestamp '{timestamp}' and model '{model_name}'."
        )

    return ckpt_paths


def predict_text(
    text: str,
    timestamp: str,
    model_name: str,
    device: str,
    fold_index: int | None = None,
) -> float:
    frame = pd.DataFrame({"excerpt": [text]})
    return predict_frame(frame, timestamp, model_name, device, fold_index=fold_index)[0]


def predict_frame(
    frame: pd.DataFrame,
    timestamp: str,
    model_name: str,
    device: str,
    fold_index: int | None = None,
) -> list[float]:
    ckpt_paths = _collect_ckpt_paths(timestamp, model_name, fold_index=fold_index)

    predictions = []
    for ckpt_path in ckpt_paths:
        model, tokenizer = load_checkpoint_bundle(ckpt_path)
        dataset = CommonLitDataset(frame, tokenizer)
        loader = DataLoader(dataset, batch_size=1, num_workers=0)

        model.to(device)
        model.eval()

        with torch.no_grad():
            fold_predictions = []
            for inputs, _, features in loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                mean, _ = model(features.to(device), **inputs)
                fold_predictions.append(mean.squeeze().cpu())
            predictions.append(torch.stack(fold_predictions).numpy())

    return list(sum(predictions) / len(predictions))


def predict_csv(
    csv_path: Path,
    timestamp: str,
    model_name: str,
    device: str,
    fold_index: int | None = None,
    text_column: str = "excerpt",
    id_column: str = "id",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise KeyError(f"Missing text column '{text_column}' in {csv_path}")

    preds = predict_frame(
        df[[text_column]].rename(columns={text_column: "excerpt"}),
        timestamp,
        model_name,
        device,
        fold_index=fold_index,
    )
    out = pd.DataFrame({"target": preds})

    if id_column in df.columns:
        out.insert(0, id_column, df[id_column])

    return out


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to score. Omit when using --csv.",
    )
    parser.add_argument(
        "--timestamp",
        default="latest",
        help="Run timestamp to load checkpoints from, or 'latest'",
    )
    parser.add_argument(
        "--model-name",
        default="roberta-base",
        help="Model directory name under the timestamp folder",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, mps, cuda, cpu",
    )
    parser.add_argument(
        "--csv",
        help="Optional CSV file to score, such as input/test.csv",
    )
    parser.add_argument(
        "--output",
        help="Optional output CSV path for predictions",
    )
    parser.add_argument(
        "--text-column",
        default="excerpt",
        help="Column containing the text to score in CSV mode",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="ID column to preserve in CSV mode",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="Optional single fold index to use instead of averaging all folds",
    )

    args = parser.parse_args()

    timestamp = args.timestamp
    if timestamp == "latest":
        timestamp = find_latest_timestamp(args.model_name)

    device = get_device(args.device)

    if args.csv:
        csv_path = Path(args.csv)
        out_df = predict_csv(
            csv_path,
            timestamp,
            args.model_name,
            device,
            fold_index=args.fold_index,
            text_column=args.text_column,
            id_column=args.id_column,
        )
        if args.output:
            out_path = Path(args.output)
            out_df.to_csv(out_path, index=False)
            print(f"Saved predictions to {out_path}")
        else:
            print(out_df.head().to_string(index=False))
    else:
        if not args.text:
            raise SystemExit("Provide either a text argument or --csv.")
        score = predict_text(
            args.text,
            timestamp,
            args.model_name,
            device,
            fold_index=args.fold_index,
        )
        print(f"Prediction: {score:.6f}")


if __name__ == "__main__":
    main()