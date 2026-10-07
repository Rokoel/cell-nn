"""Calibrated enclosed volumes of the same triangle surfaces used for distances."""
import numpy as np
from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkPolyDataNormals, vtkTriangleFilter, vtkMassProperties, vtkFeatureEdges

from .distances import SurfaceLocator


def surface_volume(surface):
    """Reject open/non-manifold meshes; orient faces consistently before integration."""
    clean = vtkCleanPolyData()
    clean.SetInputData(SurfaceLocator(surface).mesh)
    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(clean.GetOutputPort())
    triangles.PassLinesOff()
    triangles.PassVertsOff()
    triangles.Update()
    # Degenerate Imaris pole triangles can collapse to zero-area lines during
    # cleaning/triangulation. Keep only actual polygons for topology/integration.
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    mesh = vtkPolyData()
    mesh.SetPoints(triangles.GetOutput().GetPoints())
    mesh.SetPolys(triangles.GetOutput().GetPolys())
    edges = vtkFeatureEdges()
    edges.SetInputData(mesh)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    if edges.GetOutput().GetNumberOfCells():
        return None
    normals = vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    mass = vtkMassProperties()
    mass.SetInputConnection(normals.GetOutputPort())
    mass.Update()
    value = float(mass.GetVolume())
    return value if np.isfinite(value) and value > 0 else None


def add_cell_volumes(report, surfaces, labels=None, spacing=None):
    """Persist one row per cell. Primary volume measures the exported closed mesh."""
    rows = []
    for surface in surfaces:
        cell = int(surface.name.rsplit(' ', 1)[-1])
        volume = surface_volume(surface)
        row = dict(cell_id=cell, volume_um3=volume,
                   volume_status='closed_mesh' if volume is not None else 'invalid_open_or_nonmanifold_mesh')
        if labels is not None:
            row['voxel_volume_um3'] = float(np.count_nonzero(labels == cell) * np.prod(spacing))
            box = np.argwhere(labels == cell)
            row['cell_at_image_edge'] = bool(len(box) and (np.any(box.min(0) == 0) or np.any(box.max(0) == np.asarray(labels.shape)-1)))
        rows.append(row)
    report['cell_volumes'] = rows
    report['volume_method'] = 'enclosed_closed_triangle_mesh_um3'
    return report
