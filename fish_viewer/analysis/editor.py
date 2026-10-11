"""Minimal annotation selection and calibrated 2D point editing controls."""

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from ..views import mesh_for
from .annotations import AnnotationDocument, origin


class Handle(QtWidgets.QGraphicsEllipseItem):
    HIT_RADIUS = 12  # Screen pixels; the visible dot remains 8 pixels across.

    def __init__(self, editor, vertex, xyz):
        super().__init__(-4, -4, 8, 8)
        self.editor, self.vertex, self.xyz = editor, vertex, np.array(xyz)
        self.setFlags(
            self.GraphicsItemFlag.ItemIsMovable
            | self.GraphicsItemFlag.ItemIgnoresTransformations
        )
        self.setBrush(QtGui.QColor("#ffffff"))
        self.setPen(QtGui.QPen(QtGui.QColor("#205e58")))
        self.setZValue(20)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        u, v = editor.window.slice_view.projection_axes
        self.setPos(float(xyz[u]), float(xyz[v]))
        self.setToolTip(
            f"Point {vertex + 1}: drag to move; select then Delete point"
            if vertex is not None
            else "Slice contour: drag to move this border point"
        )

    def boundingRect(self):
        radius = self.HIT_RADIUS
        return QtCore.QRectF(-radius, -radius, 2 * radius, 2 * radius)

    def shape(self):
        path = QtGui.QPainterPath()
        path.addEllipse(self.boundingRect())
        return path

    def mousePressEvent(self, event):
        self.selection = self.editor.selection
        self.axis = 2 - self.editor.window.slice_view.axis
        self.drag_start = event.scenePos()
        self.initial_position = self.pos()
        if self.vertex is not None:
            self.editor.vertex.setValue(self.vertex + 1)
        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        # Use scene coordinates explicitly so zoom and image panning cannot
        # turn a point drag into a view drag. The slice coordinate stays fixed.
        self.setPos(self.initial_position + event.scenePos() - self.drag_start)
        xyz = self.xyz.copy()
        u, v = self.editor.window.slice_view.projection_axes
        xyz[u], xyz[v] = self.pos().x(), self.pos().y()
        self.editor.preview_point(self.selection, self.vertex, self.xyz, xyz)
        event.accept()

    def mouseReleaseEvent(self, event):
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        event.accept()
        u, v = self.editor.window.slice_view.projection_axes
        xyz = self.xyz.copy()
        xyz[u] = self.pos().x()
        xyz[v] = self.pos().y()
        if not np.array_equal(xyz, self.xyz):
            # Rebuilding overlays must not destroy this item during its own event.
            editor, vertex = self.editor, self.vertex
            if self.selection[0] == "surface":
                original = self.xyz.copy()
                selection, axis = self.selection, self.axis
                QtCore.QTimer.singleShot(
                    0,
                    lambda: editor.perform(
                        lambda: editor.document.move_section(
                            selection[1], original, xyz, axis=axis
                        )
                    ),
                )
            else:
                QtCore.QTimer.singleShot(0, lambda: editor.move_point(vertex, xyz))


class AnnotationEditor(QtWidgets.QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.document = None
        self.selection = None
        self.handles = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        row = QtWidgets.QHBoxLayout()
        self.manual = QtWidgets.QCheckBox("Manual")
        self.program = QtWidgets.QCheckBox("Program")
        for box in (self.manual, self.program):
            box.setChecked(True)
            box.toggled.connect(self.visibility)
            row.addWidget(box)
        layout.addLayout(row)
        self.warning = QtWidgets.QLabel()
        self.warning.setWordWrap(True)
        self.warning.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.warning)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Annotation", "Volume / distance"])
        self.tree.setMaximumHeight(170)
        self.tree.currentItemChanged.connect(self.select)
        layout.addWidget(self.tree)
        row = QtWidgets.QHBoxLayout()
        delete = QtWidgets.QPushButton("Delete selected")
        delete.clicked.connect(self.delete)
        row.addWidget(delete)
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.undo_button.clicked.connect(
            lambda: self.document.undo.undo() if self.document else None
        )
        row.addWidget(self.undo_button)
        layout.addLayout(row)
        self.edit = QtWidgets.QCheckBox("Edit points (drag white dots)")
        self.edit.setChecked(True)
        self.edit.toggled.connect(self.draw_handles)
        layout.addWidget(self.edit)
        self.tool = QtWidgets.QComboBox()
        self.tool.addItems(["Select / move", "Add contour point", "Add signal"])
        layout.addWidget(self.tool)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Point"))
        self.vertex = QtWidgets.QSpinBox()
        self.vertex.setMinimum(1)
        self.vertex.valueChanged.connect(self.coordinates)
        row.addWidget(self.vertex)
        remove = QtWidgets.QPushButton("Delete point")
        remove.clicked.connect(self.delete_point)
        row.addWidget(remove)
        layout.addLayout(row)
        positions = QtWidgets.QFormLayout()
        self.xyz = []
        for axis in "XYZ":
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(4)
            self.xyz.append(spin)
            positions.addRow(axis, spin)
        layout.addLayout(positions)
        move = QtWidgets.QPushButton("Apply point position")
        move.clicked.connect(
            lambda: self.move_point(
                self.vertex.value() - 1, [s.value() for s in self.xyz]
            )
        )
        layout.addWidget(move)
        self.signal_channel = QtWidgets.QComboBox()
        self.signal_channel.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.signal_channel.setMinimumContentsLength(12)
        layout.addWidget(self.signal_channel)
        self.export = QtWidgets.QPushButton("Export edited Imaris…")
        self.export.clicked.connect(self.export_scene)
        layout.addWidget(self.export)
        info = QtWidgets.QLabel(
            "Select a contour and drag its white dots on the current slice. Ctrl+wheel zooms; drag empty space to pan. B points follow the nearest cell border automatically. Edited objects become Program annotations."
        )
        info.setWordWrap(True)
        info.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
        )
        layout.addWidget(info)
        self.shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, window)
        self.shortcut.activated.connect(
            lambda: (
                self.document.undo.undo()
                if self.document and not window.running and not window.loading
                else None
            )
        )
        self.delete_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), window
        )
        self.delete_shortcut.setParent(self.tree)
        self.delete_shortcut.setContext(
            QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.delete_shortcut.activated.connect(self.delete)
        self.image_delete_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), window.slice_view
        )
        self.image_delete_shortcut.setContext(
            QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.image_delete_shortcut.activated.connect(self.delete)
        window.slice_view.viewport().installEventFilter(self)
        window.slice_view.overlays_updated.connect(self.draw_handles)

    def set_dataset(self, dataset, report, cached=None):
        self.selection = None
        self.warning.setText(
            "\n".join(w for w in dataset.scene.warnings if "no saved" not in w.lower())
        )
        self.warning.setVisible(bool(self.warning.text()))
        self.document = cached or AnnotationDocument(dataset, report, self.changed)
        self.document.dataset = dataset
        self.document.changed = self.changed
        self.document.undo.canUndoChanged.connect(self.undo_button.setEnabled)
        self.signal_channel.clear()
        for i, c in enumerate(dataset.metadata.channels):
            self.signal_channel.addItem(f"New signal: channel {i + 1} · {c.name}", i)
        roles = self.window.channel_roles.get(str(self.window._source_image), {})
        signals = roles.get("signals")
        if signals:
            self.signal_channel.setCurrentIndex(
                self.signal_channel.findData(signals[0])
            )
        self.changed()

    def changed(self):
        w = self.window
        view = w.slice_view
        view.meshes = [mesh_for(s) for s in w.dataset.scene.surfaces]
        view._cuts.clear()
        self.tree.blockSignals(True)
        self.tree.clear()
        for category in ("Manual", "Program"):
            parent = QtWidgets.QTreeWidgetItem([category])
            self.tree.addTopLevelItem(parent)
            parent.setExpanded(True)
            for kind, objects in [
                ("surface", w.dataset.scene.surfaces),
                ("contour", w.dataset.scene.contours),
                ("measurement", w.dataset.scene.measurements),
            ]:
                for i, obj in enumerate(objects):
                    if origin(obj) != category:
                        continue
                    value = (
                        self.document.volumes[i]
                        if kind == "surface"
                        else self.document.distances.get(i)
                        if kind == "measurement"
                        else None
                    )
                    label = (
                        f"{value:.4f} " + ("µm³" if kind == "surface" else "µm")
                        if value is not None
                        else ""
                    )
                    item = QtWidgets.QTreeWidgetItem([obj.name, label])
                    item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (kind, i))
                    parent.addChild(item)
                    if (kind, i) == self.selection:
                        self.tree.setCurrentItem(item)
        self.tree.blockSignals(False)
        if self.selection:
            kind, index = self.selection
            objects = getattr(
                w.dataset.scene,
                {
                    "surface": "surfaces",
                    "contour": "contours",
                    "measurement": "measurements",
                }[kind],
            )
            if index >= len(objects):
                self.selection = None
        view.selected = self.selection
        if w.volume_view and w._volume_dataset is w.dataset:
            camera = w.volume_view.plotter.camera_position
            w.volume_view.set_dataset(w.dataset)
            w.volume_view.plotter.camera_position = camera
            w.volume_view.selected = self.selection
        self.visibility()
        self.coordinates()
        self.undo_button.setEnabled(self.document.undo.canUndo())
        w.excel_button.setEnabled(not w.loading and not w.running)
        self.update_tables()

    def update_tables(self):
        w = self.window
        report = self.document.report()
        w.distances.blockSignals(True)
        w.distances.setRowCount(0)
        w.counts.setRowCount(0)
        for source, points in [
            ("Manual", report["manual_measurements"]["points"]),
            ("Program", report["points"]),
        ]:
            for p in points:
                row = w.distances.rowCount()
                w.distances.insertRow(row)
                values = [
                    source + " · " + p["signal_id"],
                    p.get("cell_id"),
                    p["channel"] + 1 if p["channel"] is not None else "",
                    p.get("distance_um"),
                    p.get("score"),
                    *p["xyz"],
                    *p.get("border_xyz", ["", "", ""]),
                    "",
                ]
                for j, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(
                        ""
                        if value is None
                        else f"{value:.4f}"
                        if isinstance(value, float)
                        else str(value)
                    )
                    if j == 0:
                        item.setData(
                            QtCore.Qt.ItemDataRole.UserRole,
                            dict(p, annotation_name=p["signal_id"]),
                        )
                    w.distances.setItem(row, j, item)
        for source, cells in [
            ("Manual", report["manual_measurements"]["cells"]),
            ("Program", report["cell_volumes"]),
        ]:
            for cell in cells:
                row = w.counts.rowCount()
                w.counts.insertRow(row)
                count = sum(
                    p.get("cell_id") == cell["cell_id"]
                    for points in (
                        report["points"],
                        report["manual_measurements"]["points"],
                    )
                    for p in points
                )
                for j, value in enumerate(
                    [
                        source + " · " + cell["cell_id"],
                        "",
                        "",
                        count,
                        cell["volume_um3"],
                    ]
                ):
                    w.counts.setItem(
                        row,
                        j,
                        QtWidgets.QTableWidgetItem(
                            ""
                            if value is None
                            else f"{value:.4f}"
                            if isinstance(value, float)
                            else str(value)
                        ),
                    )
        w.distances.blockSignals(False)
        w.update_results_overview(report)

    def visibility(self):
        if not self.document:
            return
        for objects in (
            self.window.dataset.scene.surfaces,
            self.window.dataset.scene.measurements,
            self.window.dataset.scene.contours,
        ):
            for obj in objects:
                obj.visible = (
                    self.manual.isChecked()
                    if origin(obj) == "Manual"
                    else self.program.isChecked()
                )
        self.window.refresh_annotations()

    def select(self, item, previous=None):
        self.selection = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        self.window.slice_view.selected = self.selection
        if self.window.volume_view:
            self.window.volume_view.selected = self.selection
        self.coordinates()
        self.window.refresh_annotations()

    def points(self):
        if not self.selection:
            return np.empty((0, 3))
        kind, index = self.selection
        scene = self.window.dataset.scene
        obj = getattr(
            scene,
            {
                "surface": "surfaces",
                "contour": "contours",
                "measurement": "measurements",
            }[kind],
        )[index]
        return (
            obj.vertices
            if kind == "surface"
            else obj.vertices(self.window.dataset.metadata)
            if kind == "contour"
            else obj.points
        )

    def coordinates(self):
        points = self.points()
        self.vertex.blockSignals(True)
        self.vertex.setMaximum(max(1, len(points)))
        self.vertex.blockSignals(False)
        if len(points):
            for spin, value in zip(self.xyz, points[self.vertex.value() - 1]):
                spin.setValue(float(value))

    def draw_handles(self):
        for item in self.handles:
            if isValid(item) and item.scene() is not None:
                item.scene().removeItem(item)
        self.handles = []
        if not self.document or not self.edit.isChecked() or not self.selection:
            return
        view = self.window.slice_view
        points = self.points()
        axis = 2 - view.axis
        kind, index = self.selection
        scene = self.window.dataset.scene
        obj = getattr(
            scene,
            {
                "surface": "surfaces",
                "contour": "contours",
                "measurement": "measurements",
            }[kind],
        )[index]
        if not obj.visible:
            return
        if kind != "measurement" and not view.surfaces_visible:
            return
        if kind == "surface":
            if obj.time != self.window.dataset.time:
                return
            segments = view._cuts.get(
                (self.selection[1], view.axis, view.index), np.empty((0, 2, 3))
            )
            section = np.unique(np.round(segments.reshape(-1, 3), 7), axis=0)
            for xyz in section[:: max(1, int(np.ceil(len(section) / 400)))]:
                # Only the actual intersection with this plane is editable.
                # Nearby mesh vertices project to the same screen locations
                # but belong to other layers and must not cover these handles.
                distances = np.linalg.norm(points - xyz, axis=1)
                nearest = int(np.argmin(distances)) if len(distances) else None
                vertex = (
                    nearest
                    if nearest is not None and distances[nearest] < 1e-7
                    else None
                )
                xyz[axis] = view.coordinate
                item = Handle(self, vertex, xyz)
                view.scene().addItem(item)
                self.handles.append(item)
            return
        if kind == "contour":
            if obj.axis != axis or self.window.dataset.time != 0:
                return
            ids = np.flatnonzero(abs(points[:, axis] - view.coordinate) < 1e-7)
        else:
            if not view.markers_visible:
                return
            ids = np.flatnonzero(
                abs(points[:, axis] - view.coordinate)
                <= self.window.dataset.spacing[axis] / 2
            )
        for i in ids:
            if kind == "measurement" and (
                obj.labels[i].upper() == "B"
                or (not obj.all_times and obj.times[i] != self.window.dataset.time)
            ):
                continue
            item = Handle(self, int(i), points[i])
            view.scene().addItem(item)
            self.handles.append(item)

    def perform(self, callback):
        if not self.document or self.window.running or self.window.loading:
            return
        try:
            callback()
        except (ValueError, IndexError) as exc:
            self.window.error(str(exc))

    def preview_point(self, selection, vertex, original, xyz):
        view = self.window.slice_view
        kind, index = selection
        if kind == "surface":
            segments = view._cuts[(index, view.axis, view.index)].copy()
            segments[np.linalg.norm(segments - original, axis=2) < 1e-6] = xyz
        elif kind == "contour":
            points = self.points().copy()
            points[vertex] = xyz
            segments = np.stack([points, np.roll(points, -1, axis=0)], axis=1)
        else:
            return
        path = QtGui.QPainterPath()
        u, v = view.projection_axes
        for a, b in segments:
            path.moveTo(float(a[u]), float(a[v]))
            path.lineTo(float(b[u]), float(b[v]))
        for item in view._overlays:
            if item.data(0) == selection:
                item.setPath(path)

    def delete(self):
        if self.selection:
            selected = self.selection
            self.selection = None
            self.perform(lambda: self.document.delete(*selected))

    def delete_point(self):
        if self.selection:
            if self.selection[0] == "measurement":
                self.delete()
            else:
                self.perform(
                    lambda: self.document.delete_vertex(
                        *self.selection, self.vertex.value() - 1
                    )
                )

    def move_point(self, index, xyz):
        if self.selection:
            self.perform(lambda: self.document.move(*self.selection, index, xyz))

    def eventFilter(self, obj, event):
        view = self.window.slice_view
        if (
            self.document
            and not self.window.running
            and not self.window.loading
            and event.type() == QtCore.QEvent.Type.MouseButtonPress
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            pos = view.mapToScene(event.position().toPoint())
            xyz = np.zeros(3)
            u, v = view.projection_axes
            xyz[u] = pos.x()
            xyz[v] = pos.y()
            xyz[2 - view.axis] = view.coordinate
            position = event.position().toPoint()
            handles = [
                item for item in view.items(position) if isinstance(item, Handle)
            ]
            if handles:
                # Expanded hit areas may overlap. Give the closest dot priority,
                # rather than whichever handle happened to be drawn last.
                def distance(item):
                    delta = view.mapFromScene(item.scenePos()) - position
                    return delta.x() ** 2 + delta.y() ** 2

                closest = min(handles, key=distance)
                for handle in self.handles:
                    handle.setZValue(21 if handle is closest else 20)
                view.setFocus()
                return False
            item = view.itemAt(position)
            if self.edit.isChecked() and self.tool.currentIndex() == 2:
                self.perform(
                    lambda: self.document.add_signal(
                        xyz, self.signal_channel.currentData()
                    )
                )
                return True
            if (
                self.edit.isChecked()
                and self.tool.currentIndex() == 1
                and self.selection
                and self.selection[0] != "measurement"
            ):
                self.perform(lambda: self.document.add_vertex(*self.selection, xyz))
                return True
            if item and item.data(0):
                kind, index = item.data(0)
                for top in range(self.tree.topLevelItemCount()):
                    parent = self.tree.topLevelItem(top)
                    for j in range(parent.childCount()):
                        child = parent.child(j)
                        if child.data(0, QtCore.Qt.ItemDataRole.UserRole) == (
                            kind,
                            index,
                        ):
                            self.tree.setCurrentItem(child)
                            return True
        return super().eventFilter(obj, event)

    def export_scene(self):
        if not self.document:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Export current annotations", "", "Imaris image (*.ims)"
        )
        if path:
            from ..workflow.edit_export import export_document

            self.window.loading = True
            self.window.sync_buttons()

            def done(result):
                self.window.loading = False
                self.window.sync_buttons()
                self.window.phase.setText("Edited annotations exported: " + result)

            def failed(message):
                self.window.loading = False
                self.window.sync_buttons()
                self.window.error(message)

            document = self.document
            self.window.background(
                lambda _: export_document(document, path), done, failed
            )
