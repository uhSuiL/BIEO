import os
import colorsys
import numpy as np
from torch import Tensor
import zarr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_hex
import warnings

warnings.filterwarnings('ignore')


def load_em_crop(file_domain, dataset, crop, scale='s0') -> np.ndarray:
    """Load the raw EM patch corresponding to a labeled crop at a given scale.

    Reads spatial metadata (translation, voxel size) from the crop's first available
    class (all classes in a crop share the same spatial coordinates), finds the
    best-matching EM resolution level, and extracts the aligned patch.

    Args:
        file_domain: Path to the dataset .zarr store (e.g. '.../jrc_mus-liver.zarr').
        crop:        Crop name, e.g. 'crop124'.
        scale:       Label scale level, e.g. 's0', 's1'.

    Returns:
        float32 ndarray covering the same physical extent as the crop.
        Shape may differ from the label when EM resolution is coarser than the label.
    """
    file_domain = os.path.join(file_domain, f'{dataset}/{dataset}.zarr')
    store = zarr.open(file_domain, mode='r')

    # All classes in a crop share identical spatial coords; pick the first available one.
    crop_group = store[f'recon-1/labels/groundtruth/{crop}']
    any_class = next(iter(crop_group.keys()))

    label_ms = store[f'recon-1/labels/groundtruth/{crop}/{any_class}'].attrs['multiscales'][0]
    label_scale_nm = label_trans_nm = None
    for ds in label_ms['datasets']:
        if ds['path'] != scale:
            continue
        for t in ds['coordinateTransformations']:
            if t['type'] == 'scale':
                label_scale_nm = np.array(t['scale'], dtype=float)
            elif t['type'] == 'translation':
                label_trans_nm = np.array(t['translation'], dtype=float)
        break

    if label_scale_nm is None:
        raise ValueError(f"Scale level '{scale}' not found in {crop}/{any_class}")

    label_shape = store[f'recon-1/labels/groundtruth/{crop}/{any_class}/{scale}'].shape

    em_ms = store['recon-1/em/fibsem-uint8'].attrs['multiscales'][0]
    best_path = best_em_scale = best_em_trans = None
    best_diff = np.inf

    for ds in em_ms['datasets']:
        em_scale = em_trans = None
        for t in ds['coordinateTransformations']:
            if t['type'] == 'scale':
                em_scale = np.array(t['scale'], dtype=float)
            elif t['type'] == 'translation':
                em_trans = np.array(t['translation'], dtype=float)
        if em_scale is None:
            continue
        diff = float(np.linalg.norm(em_scale - label_scale_nm))
        if diff < best_diff:
            best_diff = diff
            best_path, best_em_scale, best_em_trans = ds['path'], em_scale, em_trans

    offset = np.round((label_trans_nm - best_em_trans) / best_em_scale).astype(int)

    scale_ratio = label_scale_nm / best_em_scale
    patch_shape = tuple(max(1, int(round(s * r))) for s, r in zip(label_shape, scale_ratio))

    # 5. Extract and return the patch
    em_arr = store[f'recon-1/em/fibsem-uint8/{best_path}']
    patch = em_arr[
        offset[0]: offset[0] + patch_shape[0],
        offset[1]: offset[1] + patch_shape[1],
        offset[2]: offset[2] + patch_shape[2],
    ]
    return np.array(patch, dtype=np.float32)


def load_groundtruth(data_domain, dataset, crop, class_name, scale='s0') -> np.ndarray:
    file_path = f'{dataset}/{dataset}.zarr/recon-1/labels/groundtruth/{crop}/{class_name}/{scale}'
    file_path = os.path.join(data_domain, file_path)
    arr = zarr.open(file_path, mode='r')
    arr = np.array(arr)
    return arr


def load_multiclass_groundtruth(class_names, data_domain, dataset, crop, scale='s0'):
    return np.stack([
        load_groundtruth(data_domain, dataset, crop, cls_name, scale)
        for cls_name in class_names
    ], axis=-1)


def load_result(file_domain, crop, class_name, scale='s0') -> np.ndarray:
    file_name = f'{crop}/{class_name}/{scale}'
    file_path = os.path.join(file_domain, file_name)
    data = zarr.open(file_path, mode='r')
    arr = np.array(data, dtype=np.float32)
    return arr


def load_multiclass_result(class_names, file_domain, crop, scale='s0') -> np.ndarray:
    return np.stack([
        load_result(file_domain, crop, cls_name, scale)
        for cls_name in class_names
    ], axis=-1)


def _label_palette(n_total: int) -> np.ndarray:
    """n colors evenly spaced around the hue wheel, HSV-style, but at reduced
    saturation so classes don't clash as harshly as a full-saturation HSV wheel."""
    hues = np.linspace(0, 1, n_total, endpoint=False)
    return np.array([colorsys.hsv_to_rgb(h, 0.65, 0.95) for h in hues])


def get_class_colors(label_names: list[str]) -> dict:
    """Hex color for each class name, using the same deterministic palette as
    `visualize` (so a legend built from this matches the images exactly)."""
    palette = _label_palette(len(label_names))
    return {name: to_hex(palette[i].tolist()) for i, name in enumerate(label_names)}


def visualize(pixels: Tensor, labels: Tensor, label_names: list[str] = None, ax=None, title: str = None, legend: bool = True):
    """Visualize the images with different colors based on labels

    :param pixels: (W, H), tensor of a grayscale image
    :param labels: (W, H), every pixel is matched to a label id
    :param ax: existing matplotlib Axes to draw into. If None (default), a new
        figure+axes is created. Pass an Axes from `plt.subplots(...)` to compose
        several calls into one shared canvas.
    :param title: optional title set on the axes.
    :param legend: if False, skip drawing a per-axes legend (e.g. when building
        one shared legend for a multi-panel figure via `get_class_colors`).
    """
    assert pixels.shape == labels.shape, f"Shape Mismatch: {pixels.shape, labels.shape}"
    assert pixels.dim() == labels.dim() == 2, f"Shape not allowed: {pixels.dim(), labels.dim()}"

    pixels_np = pixels.cpu().numpy()                          # (W, H)
    labels_np = labels.cpu().numpy()                          # (W, H)

    # Normalize grayscale to [0, 1]
    gray = (pixels_np - pixels_np.min()) / (pixels_np.max() - pixels_np.min() + 1e-8)

    # Build color lookup table: color is keyed by label id itself (not by its
    # position among the ids present in this image), so the same label id
    # always gets the same color across different images/calls.
    label_ids = np.unique(labels_np)
    n_total = len(label_names) if label_names is not None else int(labels_np.max()) + 1
    palette = _label_palette(n_total)  # (n_total, 3)
    id_to_color = {lid: palette[int(lid)] for lid in label_ids}

    # Map each pixel to its label color
    color_layer = np.stack([id_to_color[lid] for lid in labels_np.ravel()], axis=0)
    color_layer = color_layer.reshape(*labels_np.shape, 3)    # (W, H, 3)

    # Overlay: blend color with grayscale
    gray_rgb = np.stack([gray] * 3, axis=-1)                  # (W, H, 3)
    rgb = (0.5 * color_layer + 0.5 * gray_rgb).clip(0, 1)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.imshow(rgb.transpose(1, 0, 2), origin='lower')
    if legend:
        ax.legend(
            handles=[
                Patch(color=to_hex(id_to_color[lid].tolist()), label=f'class {int(lid)}' if label_names is None else label_names[int(lid)])
                for lid in label_ids.tolist()
            ],
            bbox_to_anchor=(1.05, 1), loc='upper left',
        )
    ax.grid(False)
    if title is not None:
        ax.set_title(title)

    if own_fig:
        plt.tight_layout()
    return fig
