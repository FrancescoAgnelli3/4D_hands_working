import argparse
import os
from typing import Optional, Tuple

from data import build_datasets, get_dataset_metadata, make_loaders
from runner import run_experiment
from train_best_models import (
    _build_dataset_cfg,
    _build_train_cfg,
    _dataset_file_path,
    _ensure_dir,
    _load_best_config,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BEST_MODELS_PATH = os.path.join(REPO_ROOT, "out", "best_runs", "assembly", "best_models.json")
DEFAULT_DATA_DIR = ""
DEFAULT_CHECKPOINT_ROOT = os.path.join(REPO_ROOT, "examples", "models")


def _normalize_model_name(name: str) -> str:
    lowered = name.strip().lower()
    if lowered == "sfgdnet":
        return "SFgDNet"
    if lowered == "splineeqnet":
        return "SplineEqNet"
    return lowered


def _default_checkpoint_path(
    model_name: str, dataset: str, action_filter: str, checkpoint_root: str
) -> str:
    action_segment = action_filter or "all"
    action_segment = action_segment.replace(os.sep, "_").replace(" ", "_")
    normalized_model = _normalize_model_name(model_name)
    model_file = f"{normalized_model}_{dataset}_{action_segment}.pt"
    return os.path.join(checkpoint_root, model_file)


def main() -> None:
    ap = argparse.ArgumentParser(description="Load best model checkpoints and evaluate on validation set only.")
    ap.add_argument(
        "--model",
        nargs="+",
        default="twostage_dct_diffusion",
        help="One or more model names (space- or comma-separated) whose best checkpoint should be evaluated.",
    )
    ap.add_argument("--dataset", default="assembly", help="Dataset to use for validation.")
    ap.add_argument("--best-json", default=BEST_MODELS_PATH, help="Path to JSON file with best configurations.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Root directory containing dataset files.")
    ap.add_argument("--action-filter", default="", help="Action substring filter for dataset files.")
    ap.add_argument("--checkpoint", default="", help="Optional explicit checkpoint path to load.")
    ap.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT, help="Root folder for saved checkpoints.")
    ap.add_argument("--seed", type=int, default=None, help="Override random seed for dataset/build pipeline.")
    ap.add_argument("--eval-batch-mult", type=int, default=1, help="Multiplier for validation batch size.")
    ap.add_argument("--save-eval-examples", action="store_true", help="When set, persist evaluation examples (merged_pred/merged_tgt tensors) during validation.")
    ap.add_argument(
        "--diffusion-best-of-k",
        type=int,
        default=8,
        help="For twostage diffusion evaluation, sample k stochastic predictions and keep the best one per sample.",
    )
    ap.add_argument("--window", type=int, default=None, help="Optional override for the input window length (input_n).")
    ap.add_argument("--horizon", type=int, default=None, help="Optional override for the prediction horizon length (output_n).")
    args = ap.parse_args()

    dataset_name = args.dataset.lower()
    metadata = get_dataset_metadata(dataset_name)

    data_dir = args.data_dir or metadata.get("default_dir", "")
    action_filter = args.action_filter or metadata.get("default_action_filter", "")
    default_wrist_indices: Tuple[int, ...] = tuple(metadata.get("default_wrist_indices", ()))
    if not default_wrist_indices:
        raise SystemExit("Dataset metadata must define default wrist indices.")
    wrist_indices = tuple(int(idx) for idx in default_wrist_indices)

    best_json_path = _dataset_file_path(args.best_json, dataset_name)
    if not os.path.exists(best_json_path) and os.path.exists(args.best_json):
        best_json_path = args.best_json

    raw_models = args.model if isinstance(args.model, list) else [str(args.model)]
    model_names = []
    for token in raw_models:
        model_names.extend([m.strip() for m in str(token).split(",") if m.strip()])

    if not model_names:
        raise SystemExit("No valid model names provided to --model")

    _ensure_dir(args.checkpoint_root)

    any_failed = False
    for model_name in model_names:
        try:
            best_cfg = _load_best_config(best_json_path, model_name)

            ds_cfg = _build_dataset_cfg(
                best_cfg,
                data_dir=data_dir,
                action_filter=action_filter,
                dataset_name=dataset_name,
                wrist_indices=wrist_indices,
                node_count=int(metadata.get("node_count", 42)),
                edge_index=tuple(metadata.get("edge_index", ())),
                adjacency=tuple(metadata.get("adjacency", ())),
                window_override=args.window,
                horizon_override=args.horizon,
                seed_override=args.seed,
                eval_batch_mult=args.eval_batch_mult,
            )

            train_cfg = _build_train_cfg(best_cfg, epochs_override=0, model_name=model_name)
            train_cfg.model = model_name
            train_cfg.epochs = 0
            train_cfg.twostage_diffusion_epochs = 0
            train_cfg.save_eval_examples = bool(args.save_eval_examples)

            train_dataset, val_dataset, test_dataset = build_datasets(ds_cfg)
            train_loader, val_loader, _ = make_loaders(
                train_dataset, val_dataset, test_dataset, ds_cfg.batch_size, ds_cfg.seed, ds_cfg.eval_batch_mult
            )

            checkpoint_path = args.checkpoint.strip()
            if not checkpoint_path:
                checkpoint_path = _default_checkpoint_path(
                    model_name,
                    ds_cfg.dataset,
                    ds_cfg.action_filter or "",
                    args.checkpoint_root,
                )
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

            model_name_clean = model_name.strip().lower()
            collect_twostage_tries = (
                model_name_clean == "twostage_dct_diffusion" and int(args.diffusion_best_of_k) > 1
            )
            metrics = run_experiment(
                ds=ds_cfg,
                train_cfg=train_cfg,
                save_root=".",
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=None,
                log_wandb=False,
                load_model_path=checkpoint_path,
                twostage_eval_phase="diffusion",
                twostage_eval_best_of_k=max(1, int(args.diffusion_best_of_k)),
                twostage_eval_collect_all=collect_twostage_tries,
            )
            if metrics is None:
                raise RuntimeError("Evaluation failed to produce metrics.")

            val_mpjpe = metrics.get("validation_mpjpe_best")
            val_mpjpe_norm = metrics.get("validation_mpjpe_norm_best")
            print(
                f"[{model_name}] validation_mpjpe={val_mpjpe} | validation_mpjpe_norm={val_mpjpe_norm} | "
                f"checkpoint={checkpoint_path}"
            )
            if collect_twostage_tries:
                per_try = metrics.get("validation_mpjpe_by_try") or []
                per_try_norm = metrics.get("validation_mpjpe_norm_by_try") or []
                if per_try:
                    for idx, mpjpe in enumerate(per_try, start=1):
                        mpjpe_norm = per_try_norm[idx - 1] if idx - 1 < len(per_try_norm) else None
                        print(
                            f"[{model_name}] try {idx}/{len(per_try)} validation_mpjpe={mpjpe} | "
                            f"validation_mpjpe_norm={mpjpe_norm}"
                        )

        except Exception as exc:
            any_failed = True
            print(f"Error while evaluating model '{model_name}': {exc}")

    if any_failed:
        raise SystemExit("One or more models failed during evaluation. Check logs above.")


if __name__ == "__main__":
    main()
