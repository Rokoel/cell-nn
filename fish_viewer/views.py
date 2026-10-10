from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor

from .models import Dataset, composite_slice


def mesh_for(surface):
    faces = np.column_stack((np.full(len(surface.triangles), 3), surface.triangles))
    return pv.PolyData(surface.vertices, faces)


def slice_segments(mesh: pv.PolyData, axis: int, position: float) -> np.ndarray:
    """Intersect actual mesh triangles with a physical coordinate plane."""
    normal = np.eye(3)[axis]
    origin = normal * position
    cut = mesh.slice(normal=normal, origin=origin)
    lines, pairs, cursor = cut.lines, [], 0
    while cursor < len(lines):
        count = int(lines[cursor])
        ids = lines[cursor + 1:cursor + count + 1]
        pairs.extend((cut.points[a], cut.points[b]) for a, b in zip(ids[:-1], ids[1:]))
        cursor += count + 1
    return np.asarray(pairs).reshape(-1, 2, 3)


def clip_segment(a, b, axis, center, half_width):
    delta = b[axis] - a[axis]
    if abs(delta) < 1e-12:
        return (a, b) if abs(a[axis] - center) <= half_width else None
    t1 = (center - half_width - a[axis]) / delta
    t2 = (center + half_width - a[axis]) / delta
    start, end = max(0, min(t1, t2)), min(1, max(t1, t2))
    return (a + start * (b - a), a + end * (b - a)) if start <= end else None


class SliceView(QtWidgets.QGraphicsView):
    stepped = QtCore.Signal(int)
    overlays_updated = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setBackgroundBrush(QtGui.QColor('#080c13'))
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setDragMode(self.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(self.ViewportAnchor.AnchorUnderMouse)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.dataset = None
        self.meshes = []
        self.axis = 0  # numpy ZYX axis
        self.index = 0
        self.selected = None
        self.surfaces_visible = self.markers_visible = self.lines_visible = True
        self._image_item = None
        self._overlays = []
        self._cuts = {}
        self._rect = QtCore.QRectF()

    def set_dataset(self, dataset: Dataset):
        self.scene().clear()
        self._image_item = None
        self._overlays = []
        self.dataset = dataset
        self.meshes = [mesh_for(s) for s in dataset.scene.surfaces]
        self._cuts.clear()
        self.index = dataset.volumes[0].shape[self.axis] // 2
        self.selected = None
        self.update_image()
        self.fit()

    @property
    def coordinate(self):
        axis_xyz = 2 - self.axis
        return self.dataset.center_origin[axis_xyz] + self.index * self.dataset.spacing[axis_xyz]

    @property
    def projection_axes(self):
        return {0: (0, 1), 1: (0, 2), 2: (1, 2)}[self.axis]

    def update_image(self):
        if self.dataset is None:
            return
        rgb = np.ascontiguousarray(composite_slice(self.dataset, self.axis, self.index))
        height, width = rgb.shape[:2]
        image = QtGui.QImage(rgb.data, width, height, rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
        if self._image_item is None:
            self._image_item = self.scene().addPixmap(QtGui.QPixmap.fromImage(image))
            self._image_item.setZValue(-10)
        else:
            self._image_item.setPixmap(QtGui.QPixmap.fromImage(image))
        u, v = self.projection_axes
        sx, sy = self.dataset.spacing[u], self.dataset.spacing[v]
        ox, oy = self.dataset.metadata.origin[u], self.dataset.metadata.origin[v]
        self._image_item.setTransform(QtGui.QTransform.fromScale(sx, sy))
        self._image_item.setPos(ox, oy)
        self._rect = QtCore.QRectF(ox, oy, width * sx, height * sy)
        self.setSceneRect(self._rect.adjusted(-sx, -sy, sx, sy))
        self.update_overlays()

    def update_overlays(self):
        if self.dataset is None:
            return
        for item in self._overlays:
            self.scene().removeItem(item)
        self._overlays.clear()
        u, v = self.projection_axes
        xyz_axis = 2 - self.axis
        coordinate = self.coordinate
        half = self.dataset.spacing[xyz_axis] / 2
        if self.surfaces_visible:
            for i, (surface, mesh) in enumerate(zip(self.dataset.scene.surfaces, self.meshes)):
                if not surface.visible or surface.time != self.dataset.time:
                    continue
                key = (i, self.axis, self.index)
                if key not in self._cuts:
                    if len(self._cuts) > 500:
                        self._cuts.clear()
                    self._cuts[key] = slice_segments(mesh, xyz_axis, coordinate)
                path = QtGui.QPainterPath()
                for a, b in self._cuts[key]:
                    path.moveTo(float(a[u]), float(a[v]))
                    path.lineTo(float(b[u]), float(b[v]))
                color = '#4ee5c3' if self.selected == ('surface', i) else QtGui.QColor.fromRgbF(*surface.color)
                pen = QtGui.QPen(color)
                pen.setWidthF(2.5 if self.selected == ('surface', i) else 1.4)
                pen.setCosmetic(True)
                item = self.scene().addPath(path, pen)
                item.setData(0, ('surface', i))
                self._overlays.append(item)
            for i, contour in enumerate(self.dataset.scene.contours):
                if not contour.visible or self.dataset.time != 0:
                    continue
                vertices = contour.vertices(self.dataset.metadata)
                path = QtGui.QPainterPath()
                for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
                    clipped = clip_segment(a, b, xyz_axis, coordinate, half)
                    if clipped is not None:
                        p, q = clipped
                        path.moveTo(float(p[u]), float(p[v]))
                        path.lineTo(float(q[u]), float(q[v]))
                pen = QtGui.QPen(QtGui.QColor('#4ee5c3') if self.selected == ('contour', i) else QtGui.QColor.fromRgbF(*contour.color))
                pen.setWidthF(2)
                pen.setCosmetic(True)
                pen.setStyle(QtCore.Qt.PenStyle.DashLine)
                item = self.scene().addPath(path, pen)
                item.setData(0, ('contour', i))
                self._overlays.append(item)
        for i, measure in enumerate(self.dataset.scene.measurements):
            if not measure.visible:
                continue
            visible = [measure.all_times or t == self.dataset.time for t in measure.times]
            color = QtGui.QColor('#4ee5c3') if self.selected == ('measurement', i) else QtGui.QColor.fromRgbF(*measure.color)
            pen = QtGui.QPen(color)
            pen.setWidthF(1.6)
            pen.setCosmetic(True)
            if self.lines_visible:
                for a, b in measure.edges:
                    if not visible[a] or not visible[b]:
                        continue
                    clipped = clip_segment(measure.points[a], measure.points[b], xyz_axis, coordinate, half)
                    if clipped is not None:
                        p, q = clipped
                        self._overlays.append(self.scene().addLine(float(p[u]), float(p[v]), float(q[u]), float(q[v]), pen))
            if self.markers_visible:
                for j, point in enumerate(measure.points):
                    if not visible[j] or abs(point[xyz_axis] - coordinate) > half:
                        continue
                    marker = QtWidgets.QGraphicsEllipseItem(-4, -4, 8, 8)
                    marker.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                    marker.setPen(pen)
                    marker.setPos(float(point[u]), float(point[v]))
                    marker.setData(0, ('measurement', i))
                    self.scene().addItem(marker)
                    self._overlays.append(marker)
                    if self.selected == ('measurement', i):
                        label = QtWidgets.QGraphicsSimpleTextItem(measure.labels[j])
                        label.setBrush(color)
                        label.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                        label.setPos(float(point[u]), float(point[v]))
                        self.scene().addItem(label)
                        self._overlays.append(label)
        self.viewport().update()
        self.overlays_updated.emit()

    def fit(self):
        if not self._rect.isEmpty():
            self.fitInView(self._rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self.scale(factor, factor)
        else:
            self.stepped.emit(1 if event.angleDelta().y() > 0 else -1)
        event.accept()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.dataset is None:
            return
        painter = QtGui.QPainter(self.viewport())
        painter.setPen(QtGui.QColor('#b1bfce'))
        painter.drawText(18, 28, f'{"ZYX"[self.axis]} = {self.coordinate:.3f} {self.dataset.metadata.unit}   ·   Slice {self.index + 1}')
        # Fixed-screen overlay; physical bar length follows the current zoom.
        pixels_per_unit = abs(self.transform().m11())
        if pixels_per_unit > 0:
            target = 100 / pixels_per_unit
            power = 10 ** np.floor(np.log10(max(target, 1e-12)))
            length = max(v * power for v in (0.1, 0.2, 0.5, 1, 2, 5, 10) if v * power <= target)
            width = int(length * pixels_per_unit)
            x, y = 22, self.viewport().height() - 30
            painter.setPen(QtGui.QPen(QtGui.QColor('#e6edf5'), 2))
            painter.drawLine(x, y, x + width, y)
            painter.drawLine(x, y - 4, x, y + 4)
            painter.drawLine(x + width, y - 4, x + width, y + 4)
            painter.drawText(x, y - 9, f'{length:g} {self.dataset.metadata.unit}')
        painter.end()


class VolumeView(QtWidgets.QWidget):
    def __init__(self, parent=None, off_screen=False):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self, off_screen=off_screen, auto_update=False)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background('#080c13')
        self.dataset = None
        self.selected = None
        self.surfaces_visible = self.markers_visible = self.lines_visible = True
        self.surface_opacity = 0.32
        self.volume_opacity = 0.25
        self.full_detail = False
        self.wireframe = False
        self._volume = None
        self._grid = None
        self._mesh_actors = []
        self._annotation_actors = []
        self._stride = 1

    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset
        self.selected = None
        self.plotter.clear()
        self._mesh_actors = []
        self._annotation_actors = []
        self._volume = None
        self._grid = None
        self.update_volume()
        for surface in dataset.scene.surfaces:
            display_mesh = mesh_for(surface).clean().smooth_taubin(n_iter=15, pass_band=.12)
            actor = self.plotter.add_mesh(display_mesh, color=surface.color,
                opacity=self.surface_opacity, smooth_shading=True, render=False,
                show_scalar_bar=False, pickable=True)
            self._mesh_actors.append(actor)
        self.update_annotations()
        self.plotter.add_axes()
        self.plotter.show_bounds(grid=False, location='outer', all_edges=False,
            xtitle=f'X ({dataset.metadata.unit})', ytitle=f'Y ({dataset.metadata.unit})',
            ztitle=f'Z ({dataset.metadata.unit})', font_size=9, color='#a6b5c8')
        self.plotter.view_isometric()
        self.fit()

    def update_volume(self):
        if self.dataset is None:
            return
        dataset = self.dataset
        strides = tuple(1 if self.full_detail else max(1, int(np.ceil(n / 256))) for n in dataset.volumes[0].shape)
        self._stride = strides
        selection = tuple(slice(None, None, stride) for stride in strides)
        shape = dataset.volumes[0][selection].shape
        rgb = np.zeros((*shape, 3), dtype=np.float32)
        for channel, volume in zip(dataset.metadata.channels, dataset.volumes):
            if channel.visible:
                intensity = channel.normalize(volume[selection])
                rgb += intensity[..., None] * np.asarray(channel.color, dtype=np.float32)
        np.clip(rgb, 0, 1, out=rgb)
        rgba = np.empty((*shape, 4), dtype=np.uint8)
        rgba[..., :3] = rgb * 255
        rgba[..., 3] = rgb.max(axis=-1) * 255
        # A single RGBA volume makes all channels participate in one ray integration.
        if self._grid is None or tuple(self._grid.dimensions) != shape[::-1]:
            if self._volume is not None:
                self.plotter.remove_actor(self._volume, render=False)
            self._grid = pv.ImageData(dimensions=shape[::-1],
                spacing=tuple(s * stride for s, stride in zip(dataset.spacing, strides[::-1])), origin=dataset.center_origin)
            self._grid.point_data['rgba'] = rgba.reshape(-1, 4)
            self._volume = self.plotter.add_volume(self._grid, scalars='rgba',
                opacity='linear', clim=(0, 255),
                shade=False, mapper='smart', blending='composite',
                opacity_unit_distance=min(dataset.spacing), show_scalar_bar=False,
                pickable=False, render=False, name='Image volume')
        else:
            self._grid.point_data['rgba'][:] = rgba.reshape(-1, 4)
            self._grid.GetPointData().GetScalars().Modified()
            self._grid.Modified()
        transfer = self._volume.GetProperty().GetScalarOpacity()
        transfer.RemoveAllPoints()
        transfer.AddPoint(0, 0)
        transfer.AddPoint(255, self.volume_opacity)
        self.plotter.render()

    def update_annotations(self):
        if self.dataset is None:
            return
        for i, (actor, surface) in enumerate(zip(self._mesh_actors, self.dataset.scene.surfaces)):
            actor.SetVisibility(self.surfaces_visible and surface.visible and surface.time == self.dataset.time)
            prop = actor.GetProperty()
            prop.SetOpacity(self.surface_opacity)
            prop.SetColor((0.3, 0.95, 0.8) if self.selected == ('surface', i) else surface.color)
            if self.wireframe:
                prop.SetRepresentationToWireframe()
            else:
                prop.SetRepresentationToSurface()
        for actor in self._annotation_actors:
            self.plotter.remove_actor(actor, render=False)
        self._annotation_actors = []
        if self.surfaces_visible and self.dataset.time == 0:
            for i, contour in enumerate(self.dataset.scene.contours):
                if contour.visible:
                    curve = pv.lines_from_points(contour.vertices(self.dataset.metadata), close=True)
                    actor = self.plotter.add_mesh(curve, color='#4ee5c3' if self.selected == ('contour', i) else contour.color,
                        line_width=2, render=False, reset_camera=False)
                    self._annotation_actors.append(actor)
        for i, measure in enumerate(self.dataset.scene.measurements):
            if not measure.visible:
                continue
            color = '#4ee5c3' if self.selected == ('measurement', i) else measure.color
            visible = np.asarray([measure.all_times or t == self.dataset.time for t in measure.times])
            if self.markers_visible and visible.any():
                actor = self.plotter.add_points(measure.points[visible], color=color, point_size=9,
                    render_points_as_spheres=True, render=False, show_scalar_bar=False, reset_camera=False)
                self._annotation_actors.append(actor)
                if self.selected == ('measurement', i) and measure.labels == ['A', 'B']:
                    actor = self.plotter.add_point_labels(measure.points[visible],
                        [text for text, shown in zip(measure.labels, visible) if shown],
                        font_size=15, text_color='#e6fff8', show_points=False,
                        shape_color='#102521', shape_opacity=.8, always_visible=True,
                        reset_camera=False, render=False)
                    self._annotation_actors.append(actor)
            if self.lines_visible:
                pairs = [(a, b) for a, b in measure.edges if visible[a] and visible[b]]
                if pairs:
                    segments = np.asarray([[measure.points[a], measure.points[b]] for a, b in pairs]).reshape(-1, 3)
                    actor = self.plotter.add_lines(segments, color=color, width=2, connected=False)
                    self._annotation_actors.append(actor)
                    if self.selected == ('measurement', i) and measure.labels == ['A', 'B']:
                        # A/B can be closer together on screen than the width of
                        # a distance label. Keep the value away from their markers.
                        actor = self.plotter.add_text(f'A-B: {measure.length:.3f} {self.dataset.metadata.unit}',
                            position='upper_left', font_size=11, color='#e6fff8',
                            name='selected-ab-distance', render=False)
                        self._annotation_actors.append(actor)
        self.plotter.render()

    def fit(self):
        if self.dataset:
            origin = self.dataset.metadata.origin
            extent = self.dataset.metadata.extent
            self.plotter.reset_camera(bounds=tuple(v for o, e in zip(origin, extent) for v in (o, o + e)))

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)
