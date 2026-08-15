"""
train_foul_classifier.py
=========================

Fine-tunes the VideoMAE model used by ``foul_classifier.py`` on the
SoccerNet-MVFoul dataset (see ``validator.py`` for the expected dataset
layout and label-normalization tables), replacing its generic Kinetics-400
action-recognition head with a direct 6-class foul-type head.

This is a two-stage fine-tune:
    Stage 1 (5 epochs): the VideoMAE encoder is frozen; only the new
        classification head trains.
    Stage 2 (5 epochs): the last 4 transformer blocks of the encoder are
        unfrozen (in addition to the head) and trained at a lower LR.

The best checkpoint (by validation macro F1, tracked across all 10 epochs)
is written to ``<output-dir>/videomae-foul-best.pt``. Once training is
done, activate it in the pipeline by following the SWAP HOOK instructions
printed at the end of this script (and mirrored in the SWAP HOOK comment
block near the top of ``foul_classifier.py``).

Usage
-----
    python train_foul_classifier.py --dataset-path ./mvfoul_dataset --output-dir ./checkpoints/
    python train_foul_classifier.py --smoke-test

``--smoke-test`` needs no dataset and no cached/downloadable model beyond
what's already resolvable in the local HF cache; it loads the real model,
replaces the head, and runs 2 real forward/backward/optimizer-step
iterations over random tensors shaped like real VideoMAE input, to verify
the training loop wiring.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# Sibling modules live alongside this file; make imports work from anywhere
# (mirrors the sys.path setup in validator.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frame_extractor import extract_frames  # noqa: E402
from validator import build_ground_truth, discover_clips  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fixed class vocabulary and index order for the fine-tuned head. This
#: exact order is what gets baked into the saved checkpoint's 'class_names'
#: and must match what foul_classifier.py expects post-swap.
CLASS_NAMES: list[str] = ["tackle", "handball", "obstruction", "simulation", "push", "none"]

#: Base checkpoint to fine-tune from -- same Kinetics-400 VideoMAE checkpoint
#: foul_classifier.py currently uses.
MODEL_ID: str = "MCG-NJU/videomae-base-finetuned-kinetics"

NUM_FRAMES: int = 16
IMG_SIZE: int = 224
NUM_TRANSFORMER_BLOCKS_TO_UNFREEZE: int = 4
EPOCHS_PER_STAGE: int = 5
TRAIN_VAL_SPLIT: float = 0.85
SPLIT_SEED: int = 42

SWAP_HOOK_INSTRUCTIONS = """\
SWAP HOOK -- activating the fine-tuned checkpoint
==================================================
To use this checkpoint in the pipeline, edit generated_pipeline/foul_classifier.py:

  1. Load the checkpoint dict from './checkpoints/videomae-foul-best.pt' via
     torch.load(...). It has keys: 'model_state_dict', 'class_names', 'epoch',
     'val_f1'.
  2. Before loading model_state_dict, replace the model's classification head
     with torch.nn.Linear(768, len(CLASS_NAMES)) (768 = VideoMAE-base hidden
     size; this must match the head shape train_foul_classifier.py trained).
  3. Use the checkpoint's 'class_names' list (index order: tackle, handball,
     obstruction, simulation, push, none) to map the argmax output index
     directly to a foul type string -- no keyword matching needed.
  4. In classify_foul, replace the call to _map_kinetics_logits_to_foul with
     a direct softmax + argmax over the 6-class logits.
  5. Delete _map_kinetics_logits_to_foul and KINETICS_LABEL_TO_FOUL_KEYWORDS
     entirely once the swap is made -- they are Kinetics-400-specific
     compatibility shims that no longer apply.
"""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _normalize_frame_count(frames: list[np.ndarray], target: int = NUM_FRAMES) -> list[np.ndarray]:
    """Pad (repeat last frame) or subsample (uniform) to exactly `target` frames.

    frame_extractor.extract_frames(..., num_frames=NUM_FRAMES) should already
    return exactly NUM_FRAMES frames, but this guards defensively in case a
    clip is short enough that padding logic elsewhere yields a different
    count.
    """
    n = len(frames)
    if n == target:
        return frames
    if n < target:
        return list(frames) + [frames[-1]] * (target - n)
    indices = [int(round(i)) for i in np.linspace(0, n - 1, target)]
    return [frames[i] for i in indices]


class FoulClipDataset(Dataset):
    """(video_path, label_index) pairs, decoded to VideoMAE-shaped tensors.

    `image_mean`/`image_std` should come from the same
    `VideoMAEImageProcessor` used by `build_model()` (see
    `VideoMAEImageProcessor.image_mean`/`.image_std`), so that pixel values
    fed to the encoder are normalized the same way the pretrained checkpoint
    expects. Defaults to VideoMAE's standard ImageNet mean/std if not given,
    but callers that already have the processor loaded should always pass
    its values explicitly.
    """

    def __init__(
        self,
        examples: list[tuple[str, int]],
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
    ):
        self.examples = examples
        mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        std = image_std if image_std is not None else [0.5, 0.5, 0.5]
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        video_path, label = self.examples[idx]
        frames, _ = extract_frames(video_path, num_frames=NUM_FRAMES)
        frames = _normalize_frame_count(frames, NUM_FRAMES)
        frame_tensors = torch.stack([self._transform(frame) for frame in frames])
        # frame_tensors: (NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE) -- matches
        # VideoMAE's expected per-sample pixel_values shape.
        return frame_tensors, label


def build_examples(dataset_path: Path) -> list[tuple[str, int]]:
    """Discover clips via validator.discover_clips/build_ground_truth and map
    each to a (video_path, label_index) pair in the CLASS_NAMES vocabulary.

    Clips whose consensus ground-truth foul_type falls outside CLASS_NAMES
    are skipped with a warning printed to stderr (should not normally
    happen since build_ground_truth's foul_type already comes from
    validator.normalize_foul_type, which only emits this vocabulary).
    """
    clips = discover_clips(dataset_path)
    examples: list[tuple[str, int]] = []
    for clip in clips:
        gt = build_ground_truth(clip["labels"])
        # Match how validator.compute_metrics scores foul_type: a clip
        # ruled "no offence" is trained/scored as "none" regardless of what
        # action_class annotators attached to it.
        foul_type = gt["foul_type"] if gt["offence"] else "none"
        if foul_type not in CLASS_NAMES:
            print(
                f"[train_foul_classifier] Skipping clip '{clip['clip_id']}': "
                f"unrecognized foul_type '{foul_type}' not in {CLASS_NAMES}.",
                file=sys.stderr,
            )
            continue
        examples.append((clip["video_path"], CLASS_NAMES.index(foul_type)))
    return examples


def split_examples(
    examples: list[tuple[str, int]], train_fraction: float = TRAIN_VAL_SPLIT, seed: int = SPLIT_SEED
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Deterministic shuffle + 85/15 train/val split.

    No existing train/val convention exists elsewhere in this repo, so this
    is a simple fixed-seed shuffle split rather than a stratified one --
    good enough for this fine-tune script; a stratified split could reduce
    per-class val-set noise if class imbalance turns out to be severe.
    """
    rng = np.random.default_rng(seed=seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    split_idx = int(round(len(shuffled) * train_fraction))
    return shuffled[:split_idx], shuffled[split_idx:]


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def build_model() -> tuple[VideoMAEForVideoClassification, VideoMAEImageProcessor]:
    """Load the base VideoMAE checkpoint, freeze the encoder, and replace the
    classification head with a fresh Linear(hidden_size, len(CLASS_NAMES)).
    """
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)

    hidden_size = getattr(model.config, "hidden_size", 768)

    # Freeze the entire encoder (all backbone params).
    for param in model.parameters():
        param.requires_grad = False

    # Replace the head; new params default to requires_grad=True.
    model.classifier = torch.nn.Linear(hidden_size, len(CLASS_NAMES))
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.config.num_labels = len(CLASS_NAMES)
    model.config.id2label = dict(enumerate(CLASS_NAMES))
    model.config.label2id = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    return model, processor


def unfreeze_last_blocks(model: VideoMAEForVideoClassification, n: int = NUM_TRANSFORMER_BLOCKS_TO_UNFREEZE) -> list[torch.nn.Parameter]:
    """Unfreeze the last `n` transformer blocks of the VideoMAE encoder.

    Returns the list of newly-unfrozen parameters (for building the stage 2
    optimizer's low-LR param group).
    """
    blocks = model.videomae.encoder.layer[-n:]
    unfrozen: list[torch.nn.Parameter] = []
    for block in blocks:
        for param in block.parameters():
            param.requires_grad = True
            unfrozen.append(param)
    return unfrozen


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def _forward_loss(model: VideoMAEForVideoClassification, pixel_values: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass + cross-entropy loss over the 6-class head.

    Logits are computed via the model's forward (not passing `labels` to
    the model itself, since its internal loss computation is keyed off the
    original config and we've swapped in a differently-shaped head) and
    scored with a plain F.cross_entropy against our own label indices.
    """
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits
    loss = F.cross_entropy(logits, labels)
    return loss, logits


def evaluate(model: VideoMAEForVideoClassification, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, float]:
    """Run validation; return (per_class_f1 array, macro_f1)."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            logits = model(pixel_values=pixel_values).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    macro_f1 = float(
        f1_score(all_labels, all_preds, average="macro", labels=list(range(len(CLASS_NAMES))), zero_division=0)
    )
    return per_class_f1, macro_f1


def _print_f1(stage: str, epoch: int, per_class_f1: np.ndarray, macro_f1: float) -> None:
    print(f"[{stage}] epoch {epoch} validation F1:")
    for name, score in zip(CLASS_NAMES, per_class_f1):
        print(f"    {name:<12} F1={score:.4f}")
    print(f"    {'macro':<12} F1={macro_f1:.4f}")


def save_checkpoint(model: VideoMAEForVideoClassification, output_dir: Path, epoch: int, val_f1: float) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "videomae-foul-best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "epoch": epoch,
            "val_f1": val_f1,
        },
        checkpoint_path,
    )
    return checkpoint_path


def train_stage(
    model: VideoMAEForVideoClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stage_name: str,
    epochs: int,
    epoch_offset: int,
    output_dir: Path,
    best_macro_f1: float,
) -> float:
    """Run `epochs` epochs of a training stage; return the (possibly
    updated) best macro F1 seen so far across all stages."""
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for local_epoch in range(1, epochs + 1):
        global_epoch = epoch_offset + local_epoch
        model.train()
        for pixel_values, labels in train_loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss, _ = _forward_loss(model, pixel_values, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
        scheduler.step()

        per_class_f1, macro_f1 = evaluate(model, val_loader, device)
        _print_f1(stage_name, global_epoch, per_class_f1, macro_f1)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            checkpoint_path = save_checkpoint(model, output_dir, global_epoch, macro_f1)
            print(f"    New best macro F1={macro_f1:.4f} -- saved checkpoint to {checkpoint_path}")

    return best_macro_f1


def run_training(dataset_path: Path, output_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_foul_classifier] Using device: {device}")

    examples = build_examples(dataset_path)
    if not examples:
        raise ValueError(
            f"No usable training examples found under '{dataset_path}'. "
            "Expected <clip_dir>/annotations.json plus one video file per clip "
            "(see validator.py's discover_clips)."
        )

    train_examples, val_examples = split_examples(examples)
    print(
        f"[train_foul_classifier] {len(examples)} clips discovered -> "
        f"{len(train_examples)} train / {len(val_examples)} val."
    )
    if not val_examples:
        warnings.warn(
            "Validation split is empty; per-epoch F1 metrics will be degenerate. "
            "Consider using a larger dataset."
        )

    model, processor = build_model()
    model.to(device)

    train_loader = DataLoader(
        FoulClipDataset(train_examples, image_mean=processor.image_mean, image_std=processor.image_std),
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        FoulClipDataset(val_examples, image_mean=processor.image_mean, image_std=processor.image_std),
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    best_macro_f1 = float("-inf")

    # Stage 1: encoder frozen, only the new head trains.
    stage1_optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4)
    best_macro_f1 = train_stage(
        model, train_loader, val_loader, stage1_optimizer, device,
        stage_name="stage1", epochs=EPOCHS_PER_STAGE, epoch_offset=0,
        output_dir=output_dir, best_macro_f1=best_macro_f1,
    )

    # Stage 2: unfreeze the last 4 transformer blocks, train alongside the head.
    unfrozen_encoder_params = unfreeze_last_blocks(model, NUM_TRANSFORMER_BLOCKS_TO_UNFREEZE)
    stage2_optimizer = torch.optim.AdamW(
        [
            {"params": unfrozen_encoder_params, "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ]
    )
    best_macro_f1 = train_stage(
        model, train_loader, val_loader, stage2_optimizer, device,
        stage_name="stage2", epochs=EPOCHS_PER_STAGE, epoch_offset=EPOCHS_PER_STAGE,
        output_dir=output_dir, best_macro_f1=best_macro_f1,
    )

    print(f"[train_foul_classifier] Training complete. Best macro F1: {best_macro_f1:.4f}")


# ---------------------------------------------------------------------------
# Smoke test (no dataset, no network required beyond resolving MODEL_ID
# from the local HF cache)
# ---------------------------------------------------------------------------


def run_smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke test] Using device: {device}")

    print(f"[smoke test] Loading model '{MODEL_ID}' and replacing head...")
    model, _processor = build_model()
    model.to(device)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4)

    rng = np.random.default_rng(seed=42)
    batch_size = 2
    for step in range(1, 3):
        pixel_values = torch.tensor(
            rng.standard_normal((batch_size, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE)), dtype=torch.float32
        ).to(device)
        labels = torch.tensor(rng.integers(0, len(CLASS_NAMES), size=batch_size), dtype=torch.long).to(device)

        optimizer.zero_grad()
        loss, logits = _forward_loss(model, pixel_values, labels)
        loss.backward()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        print(f"[smoke test] step {step}: loss={loss.item():.4f} logits.shape={tuple(logits.shape)}")

    assert logits.shape == (batch_size, len(CLASS_NAMES)), "head output shape mismatch"
    print("[smoke test] PASSED: real model load + head replacement + "
          "2 forward/backward/optimizer-step iterations succeeded.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_foul_classifier.py",
        description=(
            "Fine-tune VideoMAE on the SoccerNet-MVFoul dataset to produce a "
            "direct 6-class foul-type head for foul_classifier.py."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Root directory of the extracted MVFoul dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/",
        help="Directory to write the best checkpoint to (created if missing).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a real model load + 2 training iterations on random tensors; needs no dataset.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
    elif args.dataset_path:
        run_training(Path(args.dataset_path), Path(args.output_dir))
    else:
        parser.error("either --dataset-path or --smoke-test is required")
        return  # unreachable; parser.error exits

    print()
    print(SWAP_HOOK_INSTRUCTIONS)


if __name__ == "__main__":
    main()
