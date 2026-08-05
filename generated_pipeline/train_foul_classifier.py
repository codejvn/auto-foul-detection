"""
train_foul_classifier.py
=========================

Fine-tunes the VideoMAE model used by ``foul_classifier.py`` on the
SoccerNet-MVFoul dataset (see ``dataset_builder.py`` for the annotation
parsing, label vocabularies, and class-weight derivation), replacing its
generic Kinetics-400 action-recognition head with two direct heads on a
shared backbone: an action-class head (9-way, VARS ``ACTION_CLASSES``
order) and an offence+severity head (4-way, ``OFFENCE_SEVERITY_CLASSES``).

This is a two-stage fine-tune:
    Stage 1 (5 epochs): the VideoMAE encoder is frozen; only the two new
        classification heads train.
    Stage 2 (5 epochs): the last 4 transformer blocks of the encoder are
        unfrozen (in addition to both heads) and trained at a lower LR.

The best checkpoint (by the average of the two heads' validation macro F1,
tracked across all 10 epochs) is written to
``<output-dir>/videomae-foul-best.pt``. Once training is done, activate it
in the pipeline by following the SWAP HOOK instructions printed at the end
of this script (and mirrored in the SWAP HOOK comment block near the top of
``foul_classifier.py``).

Usage
-----
    python train_foul_classifier.py --dataset-path ./mvfoul_extracted --output-dir ./checkpoints/
    python train_foul_classifier.py --smoke-test

``--dataset-path`` is the extracted MVFoul dataset ROOT, containing
``train/``, ``valid/``, and ``test/`` subdirectories (each with its own
``annotations.json``). Training examples are built from
``<dataset-path>/train/annotations.json`` and validation examples from
``<dataset-path>/valid/annotations.json`` via ``dataset_builder.build_dataset``.

``--smoke-test`` needs no dataset and no cached/downloadable model beyond
what's already resolvable in the local HF cache; it loads the real model,
attaches both heads, and runs 2 real forward/backward/optimizer-step
iterations over random tensors shaped like real VideoMAE input (with random
labels for both heads), to verify the training loop wiring.
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

from dataset_builder import (  # noqa: E402
    ACTION_CLASSES,
    CLASS_WEIGHTS_ACTION,
    CLASS_WEIGHTS_OFFENCE_SEVERITY,
    OFFENCE_SEVERITY_CLASSES,
    build_dataset,
)
from frame_extractor import extract_frames  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Base checkpoint to fine-tune from -- same Kinetics-400 VideoMAE checkpoint
#: foul_classifier.py currently uses.
MODEL_ID: str = "MCG-NJU/videomae-base-finetuned-kinetics"

NUM_FRAMES: int = 16
IMG_SIZE: int = 224
NUM_TRANSFORMER_BLOCKS_TO_UNFREEZE: int = 4
EPOCHS_PER_STAGE: int = 5

#: Relative weighting of the two task losses in `total_loss = ACTION_LOSS_WEIGHT
#: * action_loss + SEVERITY_LOSS_WEIGHT * severity_loss`. 1:1 is a starting
#: point, not a validated choice -- tunable once both heads' standalone
#: performance is understood; kept as named module-level knobs so retuning is
#: a one-line change rather than surgery through the training loop.
ACTION_LOSS_WEIGHT: float = 1.0
SEVERITY_LOSS_WEIGHT: float = 1.0

SWAP_HOOK_INSTRUCTIONS = """\
SWAP HOOK -- activating the fine-tuned checkpoint
==================================================
To use this checkpoint in the pipeline, edit generated_pipeline/foul_classifier.py:

  1. Load the checkpoint dict from './checkpoints/videomae-foul-best.pt' via
     torch.load(...). It has keys: 'model_state_dict', 'action_classes',
     'offence_severity_classes', 'epoch', 'action_macro_f1', 'severity_macro_f1',
     'combined_macro_f1'.
  2. Before loading model_state_dict, reconstruct the dual-head architecture:
     a VideoMAEModel backbone, its pretrained fc_norm LayerNorm(768), and two
     torch.nn.Linear heads -- torch.nn.Linear(768, len(action_classes)) (9-way
     action head) and torch.nn.Linear(768, len(offence_severity_classes))
     (4-way offence+severity head), where 768 = VideoMAE-base hidden size (see
     DualHeadVideoMAE in this file). This must match the shapes
     train_foul_classifier.py trained.
  3. Use the checkpoint's 'action_classes' list (VARS order: Tackling,
     Standing tackling, High leg, Holding, Pushing, Elbowing, Challenge, Dive,
     none [reserved, index 8]) to map the action head's argmax output index to
     a foul type string, and 'offence_severity_classes' (No offence,
     Offence + No card, Offence + Yellow card, Offence + Red card) to map the
     severity head's argmax output index to a card decision.
  4. In classify_foul, replace the call to _map_kinetics_logits_to_foul with a
     direct softmax + argmax over the action head's 9-class logits (ignoring
     index 8, which is never produced by real training data).
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
    """(video_path, action_label, severity_label) triples, decoded to
    VideoMAE-shaped tensors.

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
        examples: list[dict],
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        example = self.examples[idx]
        frames, _ = extract_frames(example["video_path"], num_frames=NUM_FRAMES)
        frames = _normalize_frame_count(frames, NUM_FRAMES)
        frame_tensors = torch.stack([self._transform(frame) for frame in frames])
        # frame_tensors: (NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE) -- matches
        # VideoMAE's expected per-sample pixel_values shape.
        return frame_tensors, example["action_class_label"], example["offence_severity_class"]


def _validate_labels(examples: list[dict], split_name: str) -> None:
    """Raise loudly (never a bare assert -- python -O strips those) if any
    example carries an out-of-range label for either head."""
    for example in examples:
        action_label = example["action_class_label"]
        severity_label = example["offence_severity_class"]
        if not (0 <= action_label <= len(ACTION_CLASSES) - 1):
            raise ValueError(
                f"[{split_name}] action_id={example['action_id']!r} has out-of-range "
                f"action_class_label={action_label!r}; expected 0..{len(ACTION_CLASSES) - 1}."
            )
        if not (0 <= severity_label <= len(OFFENCE_SEVERITY_CLASSES) - 1):
            raise ValueError(
                f"[{split_name}] action_id={example['action_id']!r} has out-of-range "
                f"offence_severity_class={severity_label!r}; expected "
                f"0..{len(OFFENCE_SEVERITY_CLASSES) - 1}."
            )


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


class DualHeadVideoMAE(torch.nn.Module):
    """Shared VideoMAE backbone with two independent linear heads.

    Replicates the pooling `VideoMAEForVideoClassification.forward` applies
    before its own classifier, verified against the installed transformers
    source (`transformers.models.videomae.modeling_videomae`), not assumed:

        outputs = self.videomae(pixel_values, **kwargs)
        sequence_output = outputs.last_hidden_state
        if self.fc_norm is not None:          # true when config.use_mean_pooling
            output = sequence_output.mean(1)
            output = self.fc_norm(output)
        else:
            output = sequence_output[:, 0]

    `MCG-NJU/videomae-base-finetuned-kinetics`'s config has
    `use_mean_pooling=True` (confirmed via
    `VideoMAEConfig.from_pretrained(MODEL_ID).use_mean_pooling`), so the real
    pooling is mean-over-dim-1 THEN an `fc_norm` LayerNorm -- not mean alone.

    Loading the bare `VideoMAEModel` directly (an earlier version of this
    wrapper did this) silently drops `fc_norm`: the checkpoint's
    `fc_norm.weight`/`fc_norm.bias` keys report as UNEXPECTED because
    `VideoMAEModel` has no `fc_norm` attribute to receive them, and the two
    fresh heads would then see an unnormalized pooled vector at a different
    scale than the pretrained model ever produced. To preserve the *trained*
    fc_norm, load the full `VideoMAEForVideoClassification` checkpoint and
    keep its `.videomae` (backbone) and `.fc_norm` submodules, discarding
    only `.classifier` (which is task-specific to Kinetics-400 and being
    replaced by our two heads anyway). Confirmed empirically that this
    checkpoint's `fc_norm` is genuinely trained, not default LayerNorm init
    (weight=1, bias=0): loaded `fc_norm.weight` has mean/std
    0.6832/0.0829 (min 0.0808, max 0.8438) and `fc_norm.bias` has mean/std
    0.0083/0.0660 (min -0.8345, max 0.3442) -- see task-1-report.md for the
    verification script and full output.
    """

    def __init__(self, model_id: str = MODEL_ID):
        super().__init__()
        full_model = VideoMAEForVideoClassification.from_pretrained(model_id)
        self.videomae = full_model.videomae
        self.fc_norm = full_model.fc_norm  # pretrained LayerNorm; None if
        # config.use_mean_pooling is False for some other checkpoint -- see
        # the fallback in forward() below.
        del full_model.classifier  # Kinetics-400-specific; not reused.
        hidden_size = self.videomae.config.hidden_size
        self.action_head = torch.nn.Linear(hidden_size, len(ACTION_CLASSES))
        self.severity_head = torch.nn.Linear(hidden_size, len(OFFENCE_SEVERITY_CLASSES))

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.videomae(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state
        if self.fc_norm is not None:
            pooled = self.fc_norm(sequence_output.mean(dim=1))
        else:
            # Matches VideoMAEForVideoClassification's non-mean-pooling
            # fallback (config.use_mean_pooling=False): use the first token
            # instead of a mean-pooled+normalized vector.
            pooled = sequence_output[:, 0]
        action_logits = self.action_head(pooled)
        severity_logits = self.severity_head(pooled)
        return action_logits, severity_logits


def build_model() -> tuple[DualHeadVideoMAE, VideoMAEImageProcessor]:
    """Load the base VideoMAE checkpoint into a DualHeadVideoMAE wrapper,
    freeze the whole backbone, and leave both fresh heads trainable.

    NOTE ON HANDBALL: dataset_builder produces a `handball_label` per
    example, but no third head is added for it here. Only 21 of 2319 train
    rows are handball-positive (0.9%); a plain weighted-CE head would be
    dominated by a handful of clips and is unlikely to learn anything useful
    without oversampling or a focal-loss-style objective. That is deferred to
    a follow-up once action-class performance is understood. `handball_label`
    stays present in every example dict, simply unused by this script.
    """
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    model = DualHeadVideoMAE(MODEL_ID)

    # Freeze the entire backbone (all VideoMAE encoder/embedding params) AND
    # the pretrained fc_norm LayerNorm -- fc_norm is a trained part of the
    # backbone's pooling path (see DualHeadVideoMAE docstring), not a fresh
    # head, so it belongs in the same frozen group as the encoder. This is a
    # deliberate choice: unfreeze_last_blocks only ever unfreezes the last N
    # *transformer encoder* blocks in stage 2 (mirroring the pre-dual-head
    # script's behavior, where fc_norm was likewise never unfrozen), so
    # fc_norm stays frozen through BOTH stages unless that is revisited later.
    for param in model.videomae.parameters():
        param.requires_grad = False
    if model.fc_norm is not None:
        for param in model.fc_norm.parameters():
            param.requires_grad = False

    # Both fresh heads start trainable.
    for param in model.action_head.parameters():
        param.requires_grad = True
    for param in model.severity_head.parameters():
        param.requires_grad = True

    return model, processor


def unfreeze_last_blocks(model: DualHeadVideoMAE, n: int = NUM_TRANSFORMER_BLOCKS_TO_UNFREEZE) -> list[torch.nn.Parameter]:
    """Unfreeze the last `n` transformer blocks of the VideoMAE encoder.

    Returns the list of newly-unfrozen parameters (for building the stage 2
    optimizer's low-LR param group). `model.videomae.encoder.layer` is
    confirmed to exist on `VideoMAEModel` itself (it is not something that
    only `VideoMAEForVideoClassification` exposes) -- `VideoMAEModel` owns
    `self.encoder = VideoMAEEncoder(config)`, and `VideoMAEEncoder` owns
    `self.layer = nn.ModuleList([...])`.
    """
    blocks = model.videomae.encoder.layer[-n:]
    unfrozen: list[torch.nn.Parameter] = []
    for block in blocks:
        for param in block.parameters():
            param.requires_grad = True
            unfrozen.append(param)
    return unfrozen


def _both_head_params(model: DualHeadVideoMAE) -> list[torch.nn.Parameter]:
    return list(model.action_head.parameters()) + list(model.severity_head.parameters())


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def _forward_loss(
    model: DualHeadVideoMAE,
    pixel_values: torch.Tensor,
    action_labels: torch.Tensor,
    severity_labels: torch.Tensor,
    w_action: torch.Tensor,
    w_severity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass + weighted dual cross-entropy loss over both heads.

    Returns (total_loss, action_loss, severity_loss, action_logits,
    severity_logits) so callers can log the two loss components separately.

    The 1:1 weighting between the two task losses (ACTION_LOSS_WEIGHT /
    SEVERITY_LOSS_WEIGHT, both defaulting to 1.0) is a starting point, not a
    validated choice -- tune once both heads' standalone performance is
    understood.
    """
    action_logits, severity_logits = model(pixel_values)
    action_loss = F.cross_entropy(action_logits, action_labels, weight=w_action)
    severity_loss = F.cross_entropy(severity_logits, severity_labels, weight=w_severity)
    total_loss = ACTION_LOSS_WEIGHT * action_loss + SEVERITY_LOSS_WEIGHT * severity_loss
    return total_loss, action_loss, severity_loss, action_logits, severity_logits


def evaluate(model: DualHeadVideoMAE, loader: DataLoader, device: torch.device) -> dict:
    """Run validation; return per-head predictions/labels/F1 stats."""
    model.eval()
    action_preds: list[int] = []
    action_labels: list[int] = []
    severity_preds: list[int] = []
    severity_labels: list[int] = []
    with torch.no_grad():
        for pixel_values, a_labels, s_labels in loader:
            pixel_values = pixel_values.to(device)
            action_logits, severity_logits = model(pixel_values)
            action_preds.extend(torch.argmax(action_logits, dim=-1).cpu().tolist())
            severity_preds.extend(torch.argmax(severity_logits, dim=-1).cpu().tolist())
            action_labels.extend(a_labels.tolist())
            severity_labels.extend(s_labels.tolist())

    # Action macro F1 is computed over labels 0-7 only. Index 8 ('none') has
    # zero support by construction (dataset_builder never produces it) --
    # including it in the label set would contribute a spurious 0.0 to the
    # macro average and deflate the score by a factor of 8/9. That's an
    # artefact of the reserved slot, not a measurement of anything real.
    action_per_class_f1 = f1_score(
        action_labels, action_preds, average=None, labels=list(range(8)), zero_division=0
    )
    action_macro_f1 = float(
        f1_score(action_labels, action_preds, average="macro", labels=list(range(8)), zero_division=0)
    )

    severity_per_class_f1 = f1_score(
        severity_labels,
        severity_preds,
        average=None,
        labels=list(range(len(OFFENCE_SEVERITY_CLASSES))),
        zero_division=0,
    )
    severity_macro_f1 = float(
        f1_score(
            severity_labels,
            severity_preds,
            average="macro",
            labels=list(range(len(OFFENCE_SEVERITY_CLASSES))),
            zero_division=0,
        )
    )

    return {
        "action_per_class_f1": action_per_class_f1,
        "action_macro_f1": action_macro_f1,
        "severity_per_class_f1": severity_per_class_f1,
        "severity_macro_f1": severity_macro_f1,
    }


def _print_f1(stage: str, epoch: int, metrics: dict) -> None:
    print(f"[{stage}] epoch {epoch} validation F1:")
    print("  action head:")
    # Only classes 0-7 have per-class scores (see the macro-F1 comment in
    # evaluate()); index 8 ('none') is reserved and never scored.
    for name, score in zip(ACTION_CLASSES[:8], metrics["action_per_class_f1"]):
        print(f"    {name:<18} F1={score:.4f}")
    print(f"    {'macro (0-7)':<18} F1={metrics['action_macro_f1']:.4f}")
    print("  severity head:")
    for name, score in zip(OFFENCE_SEVERITY_CLASSES, metrics["severity_per_class_f1"]):
        print(f"    {name:<24} F1={score:.4f}")
    print(f"    {'macro':<24} F1={metrics['severity_macro_f1']:.4f}")


def save_checkpoint(
    model: DualHeadVideoMAE,
    output_dir: Path,
    epoch: int,
    action_macro_f1: float,
    severity_macro_f1: float,
    combined_macro_f1: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "videomae-foul-best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "action_classes": ACTION_CLASSES,
            "offence_severity_classes": OFFENCE_SEVERITY_CLASSES,
            "epoch": epoch,
            "action_macro_f1": action_macro_f1,
            "severity_macro_f1": severity_macro_f1,
            "combined_macro_f1": combined_macro_f1,
        },
        checkpoint_path,
    )
    return checkpoint_path


def train_stage(
    model: DualHeadVideoMAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stage_name: str,
    epochs: int,
    epoch_offset: int,
    output_dir: Path,
    best_combined_f1: float,
    w_action: torch.Tensor,
    w_severity: torch.Tensor,
) -> float:
    """Run `epochs` epochs of a training stage; return the (possibly
    updated) best combined macro F1 seen so far across all stages."""
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for local_epoch in range(1, epochs + 1):
        global_epoch = epoch_offset + local_epoch
        model.train()
        for pixel_values, action_labels, severity_labels in train_loader:
            pixel_values = pixel_values.to(device)
            action_labels = action_labels.to(device)
            severity_labels = severity_labels.to(device)

            optimizer.zero_grad()
            total_loss, action_loss, severity_loss, _, _ = _forward_loss(
                model, pixel_values, action_labels, severity_labels, w_action, w_severity
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        _print_f1(stage_name, global_epoch, metrics)

        combined_f1 = (metrics["action_macro_f1"] + metrics["severity_macro_f1"]) / 2
        print(f"    combined (avg of both macro F1) = {combined_f1:.4f}")

        if combined_f1 > best_combined_f1:
            best_combined_f1 = combined_f1
            checkpoint_path = save_checkpoint(
                model,
                output_dir,
                global_epoch,
                metrics["action_macro_f1"],
                metrics["severity_macro_f1"],
                combined_f1,
            )
            print(f"    New best combined macro F1={combined_f1:.4f} -- saved checkpoint to {checkpoint_path}")

    return best_combined_f1


def run_training(dataset_path: Path, output_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_foul_classifier] Using device: {device}")

    train_annotations = dataset_path / "train" / "annotations.json"
    valid_annotations = dataset_path / "valid" / "annotations.json"
    if not train_annotations.exists():
        raise FileNotFoundError(
            f"Expected training annotations at '{train_annotations}' but it does not exist. "
            "--dataset-path must be the extracted MVFoul root containing train/valid/test subdirectories."
        )
    if not valid_annotations.exists():
        raise FileNotFoundError(
            f"Expected validation annotations at '{valid_annotations}' but it does not exist. "
            "--dataset-path must be the extracted MVFoul root containing train/valid/test subdirectories."
        )

    # build_dataset returns (train_examples, hard_case_examples); hard cases
    # (borderline/empty severity) are reserved for calibration work, not
    # training -- only the first element is used here for both splits.
    train_examples, _ = build_dataset(train_annotations)
    val_examples, _ = build_dataset(valid_annotations)

    if not train_examples:
        raise ValueError(f"No usable training examples found in '{train_annotations}'.")

    _validate_labels(train_examples, "train")
    _validate_labels(val_examples, "valid")

    print(
        f"[train_foul_classifier] {len(train_examples)} train examples, "
        f"{len(val_examples)} valid examples."
    )
    if not val_examples:
        warnings.warn(
            "Validation split is empty; per-epoch F1 metrics will be degenerate. "
            "Consider using a larger dataset."
        )

    model, processor = build_model()
    model.to(device)

    w_action = torch.tensor(CLASS_WEIGHTS_ACTION, dtype=torch.float32, device=device)
    w_severity = torch.tensor(CLASS_WEIGHTS_OFFENCE_SEVERITY, dtype=torch.float32, device=device)

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

    best_combined_f1 = float("-inf")

    # Stage 1: encoder frozen, only the two new heads train.
    stage1_optimizer = torch.optim.AdamW(_both_head_params(model), lr=1e-4)
    best_combined_f1 = train_stage(
        model, train_loader, val_loader, stage1_optimizer, device,
        stage_name="stage1", epochs=EPOCHS_PER_STAGE, epoch_offset=0,
        output_dir=output_dir, best_combined_f1=best_combined_f1,
        w_action=w_action, w_severity=w_severity,
    )

    # Stage 2: unfreeze the last 4 transformer blocks, train alongside both heads.
    unfrozen_encoder_params = unfreeze_last_blocks(model, NUM_TRANSFORMER_BLOCKS_TO_UNFREEZE)
    stage2_optimizer = torch.optim.AdamW(
        [
            {"params": unfrozen_encoder_params, "lr": 1e-5},
            {"params": _both_head_params(model), "lr": 1e-4},
        ]
    )
    best_combined_f1 = train_stage(
        model, train_loader, val_loader, stage2_optimizer, device,
        stage_name="stage2", epochs=EPOCHS_PER_STAGE, epoch_offset=EPOCHS_PER_STAGE,
        output_dir=output_dir, best_combined_f1=best_combined_f1,
        w_action=w_action, w_severity=w_severity,
    )

    print(f"[train_foul_classifier] Training complete. Best combined macro F1: {best_combined_f1:.4f}")


# ---------------------------------------------------------------------------
# Smoke test (no dataset, no network required beyond resolving MODEL_ID
# from the local HF cache)
# ---------------------------------------------------------------------------


def run_smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke test] Using device: {device}")

    print(f"[smoke test] Loading model '{MODEL_ID}' and attaching both heads...")
    model, _processor = build_model()
    model.to(device)

    optimizer = torch.optim.AdamW(_both_head_params(model), lr=1e-4)

    w_action = torch.tensor(CLASS_WEIGHTS_ACTION, dtype=torch.float32, device=device)
    w_severity = torch.tensor(CLASS_WEIGHTS_OFFENCE_SEVERITY, dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed=42)
    batch_size = 2
    action_logits = severity_logits = None
    for step in range(1, 3):
        pixel_values = torch.tensor(
            rng.standard_normal((batch_size, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE)), dtype=torch.float32
        ).to(device)
        # Random labels in-range for each head: action in 0..7 (real rows
        # never carry the reserved index 8), severity in 0..3.
        action_labels = torch.tensor(rng.integers(0, 8, size=batch_size), dtype=torch.long).to(device)
        severity_labels = torch.tensor(
            rng.integers(0, len(OFFENCE_SEVERITY_CLASSES), size=batch_size), dtype=torch.long
        ).to(device)

        optimizer.zero_grad()
        total_loss, action_loss, severity_loss, action_logits, severity_logits = _forward_loss(
            model, pixel_values, action_labels, severity_labels, w_action, w_severity
        )
        total_loss.backward()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        print(
            f"[smoke test] step {step}: total_loss={total_loss.item():.4f} "
            f"action_loss={action_loss.item():.4f} severity_loss={severity_loss.item():.4f} "
            f"action_logits.shape={tuple(action_logits.shape)} "
            f"severity_logits.shape={tuple(severity_logits.shape)}"
        )

    expected_action_shape = (batch_size, len(ACTION_CLASSES))
    if tuple(action_logits.shape) != expected_action_shape:
        raise ValueError(
            f"action head output shape mismatch: expected {expected_action_shape}, "
            f"got {tuple(action_logits.shape)}"
        )
    expected_severity_shape = (batch_size, len(OFFENCE_SEVERITY_CLASSES))
    if tuple(severity_logits.shape) != expected_severity_shape:
        raise ValueError(
            f"severity head output shape mismatch: expected {expected_severity_shape}, "
            f"got {tuple(severity_logits.shape)}"
        )

    print("[smoke test] PASSED: real model load + dual-head attachment + "
          "2 forward/backward/optimizer-step iterations succeeded.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_foul_classifier.py",
        description=(
            "Fine-tune VideoMAE on the SoccerNet-MVFoul dataset to produce a "
            "dual-head (action-class + offence/severity) classifier for foul_classifier.py."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=(
            "Root directory of the extracted MVFoul dataset, containing "
            "train/, valid/, and test/ subdirectories (each with its own "
            "annotations.json). Training uses <dataset-path>/train/annotations.json; "
            "validation uses <dataset-path>/valid/annotations.json."
        ),
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
