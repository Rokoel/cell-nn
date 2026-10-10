"""Editable calibrated annotations, independent of inference's immutable voxel archives."""

from copy import deepcopy
import re
import numpy as np
import pyvista as pv
from PySide6 import QtGui
from ..models import Measurement
from ..views import mesh_for
from ..workflow.distances import SurfaceLocator
from ..workflow.volumes import surface_volume


def origin(obj):
    return "Manual" if getattr(obj, "source", "manual") == "manual" else "Program"


class Change(QtGui.QUndoCommand):
    def __init__(self, document, before, after, text):
        super().__init__(text)
        self.document, self.before, self.after = document, before, after

    def undo(self):
        self.document.restore(self.before)

    def redo(self):
        self.document.restore(self.after)


class AnnotationDocument:
    def __init__(self, dataset, report=None, changed=lambda: None):
        self.dataset, self.changed = dataset, changed
        self.report_template = deepcopy(report or {})
        self.undo = QtGui.QUndoStack()
        self.bindings = {}
        self.channels = {}
        self.scores = {}
        surfaces = dataset.scene.surfaces
        # Split independent native A/B pairs into individually editable signals.
        groups = []
        for m in dataset.scene.measurements:
            if (
                len(m.edges) > 1
                and all(len(set(e)) == 2 for e in m.edges)
                and len({v for e in m.edges for v in e}) == 2 * len(m.edges)
            ):
                for n, (a, b) in enumerate(m.edges):
                    copy = deepcopy(m)
                    copy.name = f"{m.name} · pair {n + 1}"
                    copy.points = m.points[[a, b]].copy()
                    copy.labels = [m.labels[a], m.labels[b]]
                    copy.times = [m.times[a], m.times[b]]
                    copy.edges = [(0, 1)]
                    groups.append(copy)
            elif len(m.points) > 1 and not m.edges:
                for n in range(len(m.points)):
                    copy = deepcopy(m)
                    copy.name = f"{m.name} · point {n + 1}"
                    copy.points = m.points[[n]].copy()
                    copy.labels = [m.labels[n]]
                    copy.times = [m.times[n]]
                    groups.append(copy)
            else:
                groups.append(m)
        dataset.scene.measurements = groups
        for i, m in enumerate(dataset.scene.measurements):
            channel = re.search(r"channel (\d+)", m.name)
            self.channels[i] = int(channel[1]) - 1 if channel else None
            score = re.search(r"score ([\d.eE+-]+)", m.name)
            self.scores[i] = float(score[1]) if score else None
            cell = re.search(r"cell (\d+)", m.name)
            if cell:
                self.bindings[i] = next(
                    (
                        j
                        for j, s in enumerate(surfaces)
                        if origin(s) == origin(m)
                        and re.search(rf"(?:Predicted )?cell {cell[1]}(?:\D|$)", s.name)
                    ),
                    None,
                )
            else:
                self.bindings[i] = None
            a = self.signal_index(m)
            if a is not None and self.bindings[i] is None:
                candidates = [
                    j
                    for j, s in enumerate(surfaces)
                    if origin(s) == origin(m) and s.time == m.times[a]
                ]
                self.bindings[i] = self.containing(m.points[a], candidates)
        manual_points = self.report_template.get("manual_measurements", {}).get(
            "points", []
        )
        for i, m in enumerate(dataset.scene.measurements):
            row = next((p for p in manual_points if p.get("signal_id") == m.name), None)
            if row is not None and self.channels[i] is None:
                self.channels[i] = row.get("channel")
        self.recalculate()

    @staticmethod
    def signal_index(m):
        return next(
            (
                i
                for i, label in enumerate(m.labels)
                if bool(re.fullmatch(r"A\d*", label.strip(), re.I))
            ),
            0 if len(m.points) == 1 and m.labels[0].strip().upper() != "B" else None,
        )

    def containing(self, point, candidates=None):
        query = pv.PolyData(np.asarray(point, float).reshape(1, 3))
        for i in (
            range(len(self.dataset.scene.surfaces))
            if candidates is None
            else candidates
        ):
            s = self.dataset.scene.surfaces[i]
            if not np.all((point >= s.vertices.min(0)) & (point <= s.vertices.max(0))):
                continue
            mesh = mesh_for(s).clean()
            if mesh.n_open_edges:
                continue
            if SurfaceLocator(s).closest(point)[1] <= 1e-7:
                return i
            if query.select_enclosed_points(mesh, tolerance=1e-7, check_surface=False)[
                "SelectedPoints"
            ][0]:
                return i
        return None

    def snapshot(self):
        return deepcopy((self.dataset.scene, self.bindings, self.channels, self.scores))

    def restore(self, state):
        self.dataset.scene, self.bindings, self.channels, self.scores = deepcopy(state)
        self.recalculate()
        self.changed()

    def apply(self, text, function):
        before = self.snapshot()
        try:
            function()
            self.recalculate()
            after = self.snapshot()
        except BaseException:
            self.dataset.scene, self.bindings, self.channels, self.scores = before
            raise
        self.undo.push(Change(self, before, after, text))

    def recalculate(self):
        self.volumes = [surface_volume(s) for s in self.dataset.scene.surfaces]
        self.distances = {}
        locators = {}
        for i, m in enumerate(self.dataset.scene.measurements):
            a = self.signal_index(m)
            cell = self.bindings.get(i)
            existing_b = next(
                (j for j, label in enumerate(m.labels) if label.upper() == "B"), None
            )
            if a is not None and existing_b is not None:
                self.distances[i] = float(
                    np.linalg.norm(m.points[existing_b] - m.points[a])
                )
            if a is None or cell is None or cell >= len(self.dataset.scene.surfaces):
                continue
            if cell not in locators:
                locators[cell] = SurfaceLocator(self.dataset.scene.surfaces[cell])
            border, distance, _ = locators[cell].closest(m.points[a])
            b = next(
                (j for j, label in enumerate(m.labels) if label.upper() == "B"), None
            )
            if (
                b is not None
                and origin(m) == "Manual"
                and origin(self.dataset.scene.surfaces[cell]) == "Manual"
            ):
                self.distances[i] = float(np.linalg.norm(m.points[b] - m.points[a]))
                continue
            if origin(self.dataset.scene.surfaces[cell]) == "Program":
                m.source = "program"
            if b is None:
                b = len(m.points)
                m.points = np.vstack([m.points, border])
                m.labels.append("B")
                m.times.append(m.times[a])
            else:
                m.points[b] = border
            m.edges = [(a, b)]
            self.distances[i] = distance

    def delete(self, kind, index):
        def change():
            if kind == "surface":
                detached = {i for i, c in self.bindings.items() if c == index}
                self.dataset.scene.surfaces.pop(index)
                self.bindings = {
                    i: None
                    if c == index
                    else c - 1
                    if c is not None and c > index
                    else c
                    for i, c in self.bindings.items()
                }
                # Keep signal centers; discard their obsolete border endpoints.
                for i, m in enumerate(self.dataset.scene.measurements):
                    if i in detached:
                        a = self.signal_index(m)
                        if a is not None:
                            m.points = m.points[[a]]
                            m.labels = ["A"]
                            m.times = [m.times[a]]
                            m.edges = []
            elif kind == "measurement":
                self.dataset.scene.measurements.pop(index)
                for attribute in ("bindings", "channels", "scores"):
                    mapping = getattr(self, attribute)
                    setattr(
                        self,
                        attribute,
                        {i - (i > index): v for i, v in mapping.items() if i != index},
                    )
            else:
                self.dataset.scene.contours.pop(index)

        self.apply("Delete " + kind, change)

    def move(self, kind, index, vertex, xyz):
        xyz = np.asarray(xyz, float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Use finite XYZ coordinates.")

        def change():
            obj = getattr(
                self.dataset.scene,
                {
                    "surface": "surfaces",
                    "measurement": "measurements",
                    "contour": "contours",
                }[kind],
            )[index]
            obj.source = "program"
            if kind == "surface":
                obj.vertices[vertex] = xyz
            elif kind == "contour":
                obj.points[vertex] = xyz[[a for a in range(3) if a != obj.axis]]
            else:
                if obj.labels[vertex].upper() == "B":
                    raise ValueError(
                        "B is calculated from A and the cell border. Edit A or the contour instead."
                    )
                obj.points[vertex] = xyz
                previous = self.bindings.get(index)
                self.bindings[index] = (
                    self.containing(xyz, [previous]) if previous is not None else None
                )
                if self.bindings[index] is None:
                    self.bindings[index] = self.containing(
                        xyz,
                        [
                            j
                            for j, s in enumerate(self.dataset.scene.surfaces)
                            if origin(s) == "Program"
                        ],
                    )
                if self.bindings[index] is None:
                    self.bindings[index] = self.containing(xyz)
                if self.bindings[index] is None:
                    obj.points = obj.points[[vertex]]
                    obj.labels = ["A"]
                    obj.times = [obj.times[vertex]]
                    obj.edges = []

        self.apply("Move point", change)

    @staticmethod
    def insert_surface_vertex(obj, xyz):
        border, _, face = SurfaceLocator(obj).closest(xyz)
        distances = np.linalg.norm(obj.vertices - border, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] < 1e-7:
            return nearest
        triangle = obj.triangles[face]
        n = len(obj.vertices)
        edge = None
        for j in range(3):
            a, b = int(triangle[j]), int(triangle[(j + 1) % 3])
            delta = obj.vertices[b] - obj.vertices[a]
            fraction = np.clip(
                np.dot(border - obj.vertices[a], delta)
                / max(np.dot(delta, delta), 1e-20),
                0,
                1,
            )
            if np.linalg.norm(obj.vertices[a] + fraction * delta - border) < 1e-7:
                edge = (a, b)
                break
        obj.vertices = np.vstack([obj.vertices, border])
        if edge is None:
            replacement = [
                [triangle[0], triangle[1], n],
                [triangle[1], triangle[2], n],
                [triangle[2], triangle[0], n],
            ]
            obj.triangles = np.vstack([np.delete(obj.triangles, face, 0), replacement])
        else:
            affected = np.array([all(v in t for v in edge) for t in obj.triangles])
            replacement = []
            for t in obj.triangles[affected]:
                j = next(
                    j for j in range(3) if {int(t[j]), int(t[(j + 1) % 3])} == set(edge)
                )
                a, b, c = t[j], t[(j + 1) % 3], t[(j + 2) % 3]
                replacement.extend([[a, n, c], [n, b, c]])
            obj.triangles = np.vstack([obj.triangles[~affected], replacement])
        return n

    def move_section(self, index, old_xyz, new_xyz):
        def change():
            obj = self.dataset.scene.surfaces[index]
            vertex = self.insert_surface_vertex(obj, old_xyz)
            obj.vertices[vertex] = new_xyz
            obj.source = "program"

        self.apply("Move slice contour point", change)

    def add_vertex(self, kind, index, xyz):
        def change():
            obj = getattr(
                self.dataset.scene, "surfaces" if kind == "surface" else "contours"
            )[index]
            obj.source = "program"
            if kind == "surface":
                self.insert_surface_vertex(obj, xyz)
            else:
                p = np.asarray(xyz)[[a for a in range(3) if a != obj.axis]]
                a = obj.points
                b = np.roll(a, -1, axis=0)
                delta = b - a
                t = np.clip(
                    np.sum((p - a) * delta, axis=1)
                    / np.maximum(np.sum(delta * delta, axis=1), 1e-20),
                    0,
                    1,
                )
                edge = np.argmin(np.linalg.norm(a + t[:, None] * delta - p, axis=1))
                obj.points = np.insert(a, edge + 1, p, axis=0)

        self.apply("Add contour point", change)

    def delete_vertex(self, kind, index, vertex):
        def change():
            obj = getattr(
                self.dataset.scene, "surfaces" if kind == "surface" else "contours"
            )[index]
            obj.source = "program"
            if kind == "contour":
                if len(obj.points) <= 3:
                    raise ValueError("A contour needs at least three points.")
                obj.points = np.delete(obj.points, vertex, 0)
                return
            if len(obj.vertices) <= 4:
                raise ValueError("A volume needs at least four vertices.")
            affected = np.any(obj.triangles == vertex, axis=1)
            edges = [tuple(t[t != vertex]) for t in obj.triangles[affected]]
            neighbors = {v for e in edges for v in e}
            if len(edges) != len(neighbors) or len(neighbors) < 3:
                raise ValueError("This point is not a removable manifold vertex.")
            chain = [edges[0][0]]
            while len(chain) < len(neighbors):
                choices = [
                    v for e in edges if chain[-1] in e for v in e if v not in chain
                ]
                if not choices:
                    raise ValueError(
                        "Cannot safely close this contour after removing the point."
                    )
                chain.append(choices[0])
            patch = pv.PolyData(
                obj.vertices[chain], np.array([len(chain), *range(len(chain))])
            ).triangulate()
            faces = np.asarray(chain)[patch.faces.reshape(-1, 4)[:, 1:]]
            triangles = np.vstack([obj.triangles[~affected], faces])
            obj.vertices = np.delete(obj.vertices, vertex, 0)
            obj.triangles = triangles - (triangles > vertex)
            if mesh_for(obj).clean().n_open_edges:
                raise ValueError("Removing this point would open the volume.")
            oriented = mesh_for(obj).compute_normals(
                consistent_normals=True, auto_orient_normals=True, split_vertices=False
            )
            obj.triangles = oriented.faces.reshape(-1, 4)[:, 1:].copy()

        self.apply("Delete contour point", change)

    def add_signal(self, xyz, channel):
        if channel is None:
            raise ValueError("Select a signal channel first.")

        def change():
            i = len(self.dataset.scene.measurements)
            m = Measurement(
                f"Signal added {i + 1}; channel {channel + 1}",
                np.asarray([xyz], float),
                ["A"],
                [self.dataset.time],
                source="program",
                color=self.dataset.metadata.channels[channel].color,
            )
            self.dataset.scene.measurements.append(m)
            self.channels[i] = channel
            self.scores[i] = None
            self.bindings[i] = self.containing(
                np.asarray(xyz),
                [
                    j
                    for j, s in enumerate(self.dataset.scene.surfaces)
                    if s.time == self.dataset.time
                ],
            )

        self.apply("Add signal", change)

    def report(self):
        result = deepcopy(self.report_template)
        for key in (
            "cell_predictions",
            "measurement",
            "signal_filter",
            "directory",
            "progress",
            "counts",
        ):
            result.pop(key, None)
        result["engine"] = "fish-edited-annotations"
        result["geometry_source"] = (
            "current editable scene; original manual A-B values preserved until editing"
        )
        result.update(
            image=str(self.dataset.metadata.path),
            channels=[{"name": c.name} for c in self.dataset.metadata.channels],
            points=[],
            cell_volumes=[],
            manual_measurements={"cells": [], "points": []},
        )
        for i, s in enumerate(self.dataset.scene.surfaces):
            row = {"cell_id": s.name, "volume_um3": self.volumes[i]}
            (
                result["manual_measurements"]["cells"]
                if origin(s) == "Manual"
                else result["cell_volumes"]
            ).append(row)
        for i, m in enumerate(self.dataset.scene.measurements):
            a = self.signal_index(m)
            if a is None:
                continue
            cell = self.bindings.get(i)
            channel = self.channels.get(i)
            row = {
                "signal_id": m.name,
                "cell_id": self.dataset.scene.surfaces[cell].name
                if cell is not None
                else None,
                "channel": channel,
                "channel_name": self.dataset.metadata.channels[channel].name
                if channel is not None
                else None,
                "xyz": m.points[a].tolist(),
                "score": self.scores.get(i),
                "distance_um": self.distances.get(i),
            }
            b = next((j for j, l in enumerate(m.labels) if l.upper() == "B"), None)
            if b is not None:
                row["border_xyz"] = m.points[b].tolist()
            (
                result["manual_measurements"]["points"]
                if origin(m) == "Manual"
                else result["points"]
            ).append(row)
        return result
