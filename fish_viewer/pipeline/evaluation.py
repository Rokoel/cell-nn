"""Object matching against incomplete annotations; unknown cells are not negatives."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from ..models import Surface
from .segmentation import Prepared, checkpoint


def rasterize_reference(surfaces: list[Surface], prepared: Prepared, time_index=0, cancel=None):
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkImagingStencil import vtkPolyDataToImageStencil, vtkImageStencilToImage
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    masks, names, skipped = [], [], []
    shape = prepared.image.shape
    shape_xyz = np.asarray(shape[::-1])
    spacing_xyz = np.asarray(prepared.spacing[::-1])
    origin = np.asarray(prepared.origin)
    for surface in surfaces:
        checkpoint(cancel)
        if surface.time != time_index:
            continue
        vertices = np.asarray(surface.vertices)
        # Stencil only a small bounding box per cell; retain a sparse ROI reference.
        lower = np.maximum(0, np.floor((vertices.min(0) - origin) / spacing_xyz).astype(int) - 1)
        upper = np.minimum(shape_xyz, np.ceil((vertices.max(0) - origin) / spacing_xyz).astype(int) + 2)
        if np.any(upper <= lower):
            skipped.append(f'{surface.name}: outside image extent')
            continue
        points = vtkPoints()
        points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices), deep=True))
        faces = np.column_stack([np.full(len(surface.triangles), 3), surface.triangles]).astype(np.int64)
        cells = vtkCellArray()
        cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(faces.ravel(), deep=True))
        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cells)
        stencil = vtkPolyDataToImageStencil()
        stencil.SetInputData(poly)
        stencil.SetOutputOrigin(*(origin + lower * spacing_xyz))
        stencil.SetOutputSpacing(*spacing_xyz)
        local_shape = upper - lower
        stencil.SetOutputWholeExtent(0, int(local_shape[0])-1, 0, int(local_shape[1])-1, 0, int(local_shape[2])-1)
        stencil.Update()
        converter = vtkImageStencilToImage()
        converter.SetInputConnection(stencil.GetOutputPort())
        converter.SetInsideValue(1)
        converter.SetOutsideValue(0)
        converter.SetOutputScalarTypeToUnsignedChar()
        converter.Update()
        mask = vtk_to_numpy(converter.GetOutput().GetPointData().GetScalars()).reshape(tuple(local_shape[::-1])).astype(bool)
        if mask.sum() < 8:
            skipped.append(f'{surface.name}: fewer than 8 reference voxels at processing resolution')
            continue
        slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower[::-1], upper[::-1]))
        masks.append((slices, mask))
        names.append(surface.name)
    return masks, names, skipped


def evaluate(labels, references, names, spacing, cancel=None):
    """Full-object IoU matching. Missed references contribute zero Dice/IoU.

    Boundary error is conditional on a match; no finite distance is fabricated
    for a missed cell. Unmatched predictions remain unscored, not false positives.
    """
    n = len(references)
    pred_sizes = np.bincount(labels.ravel())
    predicted_count = int(np.count_nonzero(pred_sizes[1:]))
    if not n:
        return dict(reference_count=0, predicted_count=predicted_count, matched_count=0,
            mean_dice=None, mean_iou=None, detection_rate=None, boundary_rmse_um=None,
            boundary_hd95_um=None, pair_mask_mse=None, objective=None,
            unscored_predictions=predicted_count, per_cell=[])
    ious = np.zeros((n, max(0, len(pred_sizes)-1)), dtype=np.float64)
    intersections = np.zeros_like(ious)
    ref_sizes = []
    for r, (slices, mask) in enumerate(references):
        checkpoint(cancel)
        overlap = np.bincount(labels[slices][mask], minlength=len(pred_sizes))[1:]
        ref_size = int(mask.sum())
        ref_sizes.append(ref_size)
        intersections[r] = overlap
        union = ref_size + pred_sizes[1:] - overlap
        ious[r] = np.divide(overlap, union, out=np.zeros_like(union, dtype=float), where=union > 0)
    matches = {}
    if ious.size:
        rows, cols = linear_sum_assignment(-ious)
        matches = {int(r): int(c)+1 for r, c in zip(rows, cols) if ious[r, c] >= 0.1}
    pred_boxes = ndi.find_objects(labels)
    per_cell = []
    for r, ((ref_box, mask), name) in enumerate(zip(references, names)):
        checkpoint(cancel)
        p = matches.get(r)
        row = dict(name=name, predicted_label=p, dice=0.0, iou=0.0, detected=False,
            boundary_rmse_um=None, boundary_hd95_um=None, pair_mask_mse=None)
        if p is not None:
            row['iou'] = float(ious[r, p-1])
            row['dice'] = float(2 * intersections[r, p-1] / (ref_sizes[r] + pred_sizes[p]))
            row['detected'] = row['iou'] >= 0.5
            pred_box = pred_boxes[p-1]
            lower = [max(0, min(a.start, b.start)-1) for a, b in zip(ref_box, pred_box)]
            upper = [min(s, max(a.stop, b.stop)+1) for a, b, s in zip(ref_box, pred_box, labels.shape)]
            box = tuple(slice(lo, hi) for lo, hi in zip(lower, upper))
            pred = labels[box] == p
            truth = np.zeros(pred.shape, bool)
            insert = tuple(slice(s.start-lo, s.stop-lo) for s, lo in zip(ref_box, lower))
            truth[insert] = mask
            a = truth & ~ndi.binary_erosion(truth)
            b = pred & ~ndi.binary_erosion(pred)
            distances = np.concatenate([ndi.distance_transform_edt(~a, sampling=spacing)[b],
                ndi.distance_transform_edt(~b, sampling=spacing)[a]])
            row['boundary_rmse_um'] = float(np.sqrt(np.mean(distances**2)))
            row['boundary_hd95_um'] = float(np.percentile(distances, 95))
            row['pair_mask_mse'] = float(np.mean(truth != pred))
        per_cell.append(row)
    def mean(key, conditional=False):
        values = [row[key] for row in per_cell if row[key] is not None]
        return float(np.mean(values)) if values else None
    dice = mean('dice')
    return dict(reference_count=n, predicted_count=predicted_count, matched_count=len(matches),
        mean_dice=dice, mean_iou=mean('iou'), detection_rate=mean('detected'),
        boundary_rmse_um=mean('boundary_rmse_um'), boundary_hd95_um=mean('boundary_hd95_um'),
        pair_mask_mse=mean('pair_mask_mse'), objective=1-dice,
        unscored_predictions=predicted_count-len(matches), per_cell=per_cell)


def aggregate(results):
    """Macro average over images: a densely annotated image cannot dominate tuning."""
    valid = [r for r in results if r.get('reference_count', 0)]
    output = {'scored_images': len(valid)}
    for key in ('mean_dice', 'mean_iou', 'detection_rate', 'boundary_rmse_um', 'boundary_hd95_um', 'pair_mask_mse', 'objective'):
        values = [r[key] for r in valid if r.get(key) is not None]
        output[key] = float(np.mean(values)) if values else None
    for key in ('reference_count', 'predicted_count', 'matched_count', 'unscored_predictions'):
        output[key] = sum(r.get(key, 0) for r in results)
    return output
