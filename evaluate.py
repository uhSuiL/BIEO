"""Batch voxel-wise multiclass accuracy evaluation over predicted crops.

Mirrors the "计算accuracy" cell in notebooks/test_sa2.ipynb
(argmax(pred) == argmax(groundtruth), averaged over voxels), with two fixes
over that cell:

1. Groundtruth channels are loaded in the exact order of `class_names`
   instead of via the positional slice `class_names[:-1]`. That slice only
   isolates 'bg' when 'bg' happens to be the last entry (true in the old
   4-class setup); with 6 classes it drops 'pm' instead and leaves a
   stale/duplicate 'bg' channel in the mix, silently misaligning
   groundtruth labels against prediction labels.

2. Predictions and groundtruth are not always stored at corresponding scale
   levels (some crops are resampled to isotropic voxels during prediction,
   which changes the z voxel count relative to the anisotropic groundtruth
   pyramid). Rather than assume a fixed `pred_scale`/`gt_scale` pair, we
   search the groundtruth scale pyramid for the level whose shape matches
   the prediction exactly, and skip the crop if none matches.
"""

from pathlib import Path

import numpy as np
import torch
import zarr
from scipy import ndimage

from post_process import util

_STRUCTURE_6CONN = ndimage.generate_binary_structure(3, 1)


class ShapeMismatchError(Exception):
    pass


def _label_group_path(domain: str, dataset: str, crop: str, class_name: str) -> Path:
    return Path(domain) / dataset / f'{dataset}.zarr' / 'recon-1' / 'labels' / 'groundtruth' / crop / class_name


def find_matching_gt_scale(
    pred_shape: tuple,
    origin_domain: str,
    bg_data_domain: str,
    dataset: str,
    crop: str,
    class_names: list[str],
    bg_label: str = 'bg',
    preferred_scale: str = 's1',
) -> str | None:
    """Find the groundtruth scale level whose array shape equals `pred_shape`.

    Checks the first class in `class_names` only (all classes in a crop
    share the same scale grid, per util.load_em_crop's assumption).
    """
    ref_name = class_names[0]
    domain = bg_data_domain if ref_name == bg_label else origin_domain
    group = zarr.open(str(_label_group_path(domain, dataset, crop, ref_name)), mode='r')

    scale_paths = [k for k in group.keys() if k.startswith('s')]
    scale_paths.sort(key=lambda s: (s != preferred_scale, s))

    for scale_path in scale_paths:
        if group[scale_path].shape == pred_shape:
            return scale_path
    return None


def load_groundtruth_multiclass_aligned(
    class_names: list[str],
    origin_domain: str,
    bg_data_domain: str,
    dataset: str,
    crop: str,
    scale: str,
    bg_label: str = 'bg',
) -> np.ndarray:
    """Load groundtruth channels in the exact order of `class_names`.

    The `bg_label` channel is read from `bg_data_domain` (the corrected
    bg-only package); every other channel is read from `origin_domain`.

    :return: (W, H, D, len(class_names))
    """
    channels = [
        util.load_groundtruth(
            bg_data_domain if name == bg_label else origin_domain,
            dataset, crop, name, scale,
        )
        for name in class_names
    ]
    return np.stack(channels, axis=-1)


def compute_iou(
    pred_labels: torch.Tensor | np.ndarray,
    gt_labels: torch.Tensor | np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """Per-class IoU (intersection over union) between two voxel label maps.

    :param pred_labels: (W, H, D) integer tensor/array of predicted class ids.
    :param gt_labels: (W, H, D) integer tensor/array of groundtruth class ids, same shape.
    :param class_names: class names indexed by label id, i.e. `class_names[i]` is the
        name of the class with id `i` (as produced by `argmax` over a `(..., len(class_names))`
        multiclass tensor).
    :return: {class_name: IoU} for every class present in `pred_labels` or `gt_labels`,
        plus 'mean_iou' (macro average over those classes). Classes absent from both
        pred and gt are skipped so they don't inflate the mean.
    :raises ShapeMismatchError: `pred_labels.shape != gt_labels.shape`.
    """
    pred_labels = torch.as_tensor(pred_labels)
    gt_labels = torch.as_tensor(gt_labels)

    if pred_labels.shape != gt_labels.shape:
        raise ShapeMismatchError(
            f"prediction shape {tuple(pred_labels.shape)} != groundtruth shape {tuple(gt_labels.shape)}"
        )

    iou_per_class = {}
    for label_id, name in enumerate(class_names):
        pred_mask = pred_labels == label_id
        gt_mask = gt_labels == label_id
        union = torch.count_nonzero(pred_mask | gt_mask)
        if union == 0:
            continue
        intersection = torch.count_nonzero(pred_mask & gt_mask)
        iou_per_class[name] = (intersection / union).item()

    iou_per_class['mean_iou'] = float(np.mean(list(iou_per_class.values()))) if iou_per_class else 0.0
    return iou_per_class


def compute_precision_recall_f1(
    pred_labels: torch.Tensor | np.ndarray,
    gt_labels: torch.Tensor | np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    """Per-class precision/recall/F1 between two voxel label maps.

    :param pred_labels: (W, H, D) integer tensor/array of predicted class ids.
    :param gt_labels: (W, H, D) integer tensor/array of groundtruth class ids, same shape.
    :param class_names: class names indexed by label id, i.e. `class_names[i]` is the
        name of the class with id `i`.
    :return: {class_name: {'precision', 'recall', 'f1'}} for every class present in
        `pred_labels` or `gt_labels`, plus a 'mean' entry (macro average over those
        classes). Classes absent from both pred and gt are skipped so they don't
        dilute the mean.
    :raises ShapeMismatchError: `pred_labels.shape != gt_labels.shape`.
    """
    pred_labels = torch.as_tensor(pred_labels)
    gt_labels = torch.as_tensor(gt_labels)

    if pred_labels.shape != gt_labels.shape:
        raise ShapeMismatchError(
            f"prediction shape {tuple(pred_labels.shape)} != groundtruth shape {tuple(gt_labels.shape)}"
        )

    per_class = {}
    for label_id, name in enumerate(class_names):
        pred_mask = pred_labels == label_id
        gt_mask = gt_labels == label_id
        if not pred_mask.any() and not gt_mask.any():
            continue

        tp = torch.count_nonzero(pred_mask & gt_mask)
        fp = torch.count_nonzero(pred_mask & ~gt_mask)
        fn = torch.count_nonzero(~pred_mask & gt_mask)

        precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[name] = {'precision': precision, 'recall': recall, 'f1': f1}

    per_class['mean'] = {
        metric: float(np.mean([v[metric] for k, v in per_class.items() if k != 'mean']))
        for metric in ('precision', 'recall', 'f1')
    } if per_class else {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    return per_class


# ─────────────────────────────────────────────
# Constraint Violation Statistics
# ─────────────────────────────────────────────

def compute_forbidden_adjacency_stats(
    labels: torch.Tensor | np.ndarray,
    forbidden_pairs: list[tuple[int, int]],
) -> dict:
    """Count forbidden label adjacencies actually occurring in a 3D label volume.

    An "adjacency" is a face-sharing (6-connected) pair of voxels. Each pair
    of neighboring voxels is counted once (only the +x/+y/+z directions are
    walked), matching the undirected convention used by
    `objective.total_cost`/`objective.adjacency_cost`.

    :param labels: (W, H, D) integer label array/tensor.
    :param forbidden_pairs: list of (l, l') label pairs to check for, order
        doesn't matter — (l, l') and (l', l) adjacencies are both counted
        under the key as given in `forbidden_pairs`.
    :return: {
        'per_pair': {(l, l'): count},  # occurrences of each forbidden pair
        'total': int,                  # sum over all pairs
    }
    """
    labels_np = np.asarray(labels.cpu() if isinstance(labels, torch.Tensor) else labels)
    assert labels_np.ndim == 3, f'labels must be 3D, got shape {labels_np.shape}'

    neighbor_pairs = [
        (labels_np[:-1, :, :], labels_np[1:, :, :]),
        (labels_np[:, :-1, :], labels_np[:, 1:, :]),
        (labels_np[:, :, :-1], labels_np[:, :, 1:]),
    ]

    per_pair = {}
    total = 0
    for (l, l_prime) in forbidden_pairs:
        count = 0
        for a, b in neighbor_pairs:
            count += int((((a == l) & (b == l_prime)) | ((a == l_prime) & (b == l))).sum())
        per_pair[(l, l_prime)] = count
        total += count

    return {'per_pair': per_pair, 'total': total}


def _touches_boundary(mask: np.ndarray) -> bool:
    """(W, H, D) bool -> whether any True voxel lies on the volume's outer shell."""
    return bool(
        mask[0, :, :].any() or mask[-1, :, :].any() or
        mask[:, 0, :].any() or mask[:, -1, :].any() or
        mask[:, :, 0].any() or mask[:, :, -1].any()
    )


def compute_forbidden_enclosure_components(
    labels: torch.Tensor | np.ndarray,
    forbidden_enclosure_pairs: list[tuple[int, int]],
) -> dict:
    """Count connected components that are strictly, illegally enclosed.

    For each (inner_layer_id, outer_layer_id) pair, finds every 6-connected
    connected component of voxels labeled `inner_layer_id` and flags it as an
    illegal enclosure iff EVERY face-neighbor voxel just outside the
    component is labeled `outer_layer_id`. This is an exact, component-level
    check — unlike `objective.enclosure_cost`, which only approximates it
    per-voxel (no connectivity) because a global check can't be
    checkerboard-parallelized cheaply enough for use inside an optimization
    loop. Components touching the volume's outer boundary are never flagged,
    since voxels beyond the crop are unknown and the component may in fact
    escape there.

    :param labels: (W, H, D) integer label array/tensor.
    :param forbidden_enclosure_pairs: list of (inner_layer_id, outer_layer_id).
    :return: {
        'components': [
            {'inner': l, 'outer': l', 'component_id': id, 'size': voxel_count}, ...
        ],  # one entry per illegal component, across all pairs
        'per_pair_count': {(inner, outer): num_illegal_components},
        'total': int,  # total number of illegal components, across all pairs
    }
    """
    labels_np = np.asarray(labels.cpu() if isinstance(labels, torch.Tensor) else labels)
    assert labels_np.ndim == 3, f'labels must be 3D, got shape {labels_np.shape}'

    illegal_components = []
    per_pair_count = {}

    for (inner_layer_id, outer_layer_id) in forbidden_enclosure_pairs:
        inner_mask = labels_np == inner_layer_id
        components, n_components = ndimage.label(inner_mask, structure=_STRUCTURE_6CONN)

        count = 0
        for comp_id in range(1, n_components + 1):
            comp_mask = components == comp_id
            if _touches_boundary(comp_mask):
                continue

            frontier = ndimage.binary_dilation(comp_mask, structure=_STRUCTURE_6CONN) & ~comp_mask
            if not frontier.any():
                continue

            if np.all(labels_np[frontier] == outer_layer_id):
                count += 1
                illegal_components.append({
                    'inner': inner_layer_id,
                    'outer': outer_layer_id,
                    'component_id': comp_id,
                    'size': int(comp_mask.sum()),
                })

        per_pair_count[(inner_layer_id, outer_layer_id)] = count

    return {
        'components': illegal_components,
        'per_pair_count': per_pair_count,
        'total': len(illegal_components),
    }


def compute_crop_accuracy(
    class_names: list[str],
    predictions_domain: str,
    origin_domain: str,
    bg_data_domain: str,
    dataset: str,
    crop: str,
    pred_scale: str = 's0',
    gt_scale: str = 's1',
) -> float:
    """Voxel-wise multiclass accuracy for a single crop: mean(argmax(pred) == argmax(gt)).

    :raises ShapeMismatchError: no groundtruth scale level matches the prediction shape.
    """
    pred_logits = util.load_multiclass_result(class_names, predictions_domain, crop, scale=pred_scale)
    pred_labels = pred_logits.argmax(axis=-1)

    matched_scale = find_matching_gt_scale(
        pred_labels.shape, origin_domain, bg_data_domain, dataset, crop, class_names,
        preferred_scale=gt_scale,
    )
    if matched_scale is None:
        raise ShapeMismatchError(
            f"no groundtruth scale matches prediction shape {pred_labels.shape}"
        )

    gt = load_groundtruth_multiclass_aligned(
        class_names, origin_domain, bg_data_domain, dataset, crop, scale=matched_scale,
    )
    gt_labels = gt.argmax(axis=-1)

    return float((pred_labels == gt_labels).mean())


def evaluate_predictions(
    predictions_domain: str,
    origin_domain: str,
    bg_data_domain: str,
    class_names: list[str],
    threshold: float = 0.7,
    pred_scale: str = 's0',
    gt_scale: str = 's1',
    verbose: bool = True,
) -> dict[str, dict]:
    """Compute per-crop accuracy for every dataset under `predictions_domain`.

    Crops with no matching groundtruth (missing labels, or no groundtruth
    scale level whose shape matches the prediction) are skipped.

    :return: {
        dataset_name: {
            'crop_accuracy': {crop_name: accuracy},
            'num_crops': int,               # crops actually scored
            'mean_accuracy': float,
            'above_threshold_ratio': float, # fraction of scored crops with accuracy > threshold
        },
        ...
    }
    """
    results = {}

    for dataset_dir in sorted(Path(predictions_domain).glob('*.zarr')):
        dataset = dataset_dir.name.removesuffix('.zarr')
        crop_names = sorted(p.name for p in dataset_dir.glob('crop*') if p.is_dir())

        crop_accuracy_map = {}
        for crop in crop_names:
            try:
                crop_accuracy_map[crop] = compute_crop_accuracy(
                    class_names, str(dataset_dir), origin_domain, bg_data_domain,
                    dataset, crop, pred_scale, gt_scale,
                )
            except (zarr.errors.PathNotFoundError, ShapeMismatchError) as e:
                if verbose:
                    print(f"  [skip] {dataset}/{crop}: {e}")

        if not crop_accuracy_map:
            continue

        accs = np.array(list(crop_accuracy_map.values()))
        results[dataset] = {
            'crop_accuracy': crop_accuracy_map,
            'num_crops': len(accs),
            'mean_accuracy': float(accs.mean()),
            'above_threshold_ratio': float((accs > threshold).mean()),
        }

        if verbose:
            r = results[dataset]
            print(
                f"{dataset}: {r['num_crops']} crops, "
                f"mean_accuracy={r['mean_accuracy']:.4f}, "
                f">{threshold:.0%} ratio={r['above_threshold_ratio']:.2%}"
            )

    return results