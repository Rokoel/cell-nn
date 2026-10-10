"""Export the live annotation document, including imported/manual geometry."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
from .export import _copy_image, _publish
from .imaris_scene import scene_xml, template, _component, _set
from .excel import write_excel
from ..analysis.annotations import origin
from ..models import LibraryEntry
from ..reader import load_dataset


def editable_scene_xml(image_name, scene, metadata):
    # Build all native image/camera settings through the established exporter.
    root = ET.fromstring(
        scene_xml(image_name, [], metadata, {"points": [], "channels": []})
    )
    target = root.find("bpSurfaceApplication/bpComponentGroup/bpSurfaceComponent")
    prototype = template().find(
        "bpSurfaceApplication/bpComponentGroup/bpSurfaceComponent/bpComponentGroup"
    )
    for i, s in enumerate(scene.surfaces):
        copy = deepcopy(s)
        copy.name = f"Predicted cell {i + 1}"
        part = ET.fromstring(
            scene_xml(image_name, [copy], metadata, {"points": [], "channels": []})
        )
        group = part.find(
            "bpSurfaceApplication/bpComponentGroup/bpSurfaceComponent/bpComponentGroup"
        )
        label = origin(s) + " · " + s.name
        _component(group, label)
        viewer = group.find("bpSurfaceComponent/bpSurfacesViewer")
        _component(viewer, label, s.color, s.opacity)
        _set(viewer, "bpSurfaces/name", label)
        _set(viewer, "bpSurfaces/historyLog", "FISH annotation origin: " + origin(s))
        target.append(group)
    for m in scene.measurements:
        group = deepcopy(prototype)
        comp = group.find("bpSurfaceComponent")
        _component(group, origin(m) + " · " + m.name)
        viewer = comp.find("bpSurfacesViewer")
        comp.remove(viewer)
        marker = comp.find("bpMeasurementPoints")
        _component(marker, origin(m) + " · " + m.name, m.color)
        for point in marker.findall("Point"):
            marker.remove(point)
        # Native XML measurement groups represent a polyline. Split disjoint
        # edge pairs into independent groups rather than joining unrelated spots.
        if m.edges == [(i, i + 1) for i in range(len(m.points) - 1)] or not m.edges:
            pairs = [tuple(range(len(m.points)))]
        else:
            pairs = m.edges
        used = {i for edge in pairs for i in edge}
        pairs = list(pairs) + [(i,) for i in range(len(m.points)) if i not in used]
        for edge in pairs:
            item = deepcopy(group)
            points = item.find("bpSurfaceComponent/bpMeasurementPoints")
            _set(points, "LineMode", "Polygon" if m.edges and len(edge) > 1 else "None")
            _set(points, "VisibleForAllTimepoints", "true" if m.all_times else "false")
            for offset, index in enumerate(edge):
                points.insert(
                    1 + offset,
                    ET.Element(
                        "Point",
                        Name=m.labels[index],
                        Position=" ".join(f"{v:.12g}" for v in m.points[index]),
                        TimeIndex=str(m.times[index]),
                    ),
                )
            target.append(item)
    for c in scene.contours:
        if c.native_xml is not None:
            manual = ET.fromstring(c.native_xml)
            component = manual.find("bpSurfaceComponent")
            name = component.find("name")
            name.text = origin(c) + " · " + c.name
            store = manual.find("store")
            for element in list(store):
                store.remove(element)
        else:
            manual = ET.Element("bpManualSurface")
            component = ET.SubElement(manual, "bpSurfaceComponent")
            ET.SubElement(component, "name").text = origin(c) + " · " + c.name
            material = ET.SubElement(component, "material")
            ET.SubElement(material, "diffuseColor").text = " ".join(
                str(v) for v in c.color
            )
            ET.SubElement(manual, "direction").text = str(c.axis)
            store = ET.SubElement(manual, "store")
        for index in range(c.plane + 1):
            plane = ET.SubElement(store, "plane")
            if index == c.plane:
                line = ET.SubElement(plane, "line")
                for x, y in c.points:
                    ET.SubElement(line, "vertex", x=f"{x:.12g}", y=f"{y:.12g}")
        target.append(manual)
    for index, component in enumerate(root.iter("bpSurfaceComponent")):
        ident = component.find("id")
        if ident is not None:
            ident.text = str(index + 1)
    ET.indent(root)
    return (
        ET.tostring(root, encoding="utf-8", short_empty_elements=False).replace(
            b"\n", b"\r\n"
        )
        + b"\r\n"
    )


def export_document(document, path):
    target = Path(path).expanduser().resolve().with_suffix(".ims")
    source = document.dataset.metadata.path.resolve()
    if source == target:
        raise ValueError("Export to a new filename; the original scan is read-only.")
    destinations = [
        target.with_suffix(s) for s in (".ims", ".imx", ".xlsx", ".annotations.json")
    ]
    if any(p.exists() for p in destinations):
        raise FileExistsError(
            "Choose a new filename; existing files are never overwritten."
        )
    # Snapshot before export so later UI edits cannot change half an export.
    scene = deepcopy(document.dataset.scene)
    report = document.report()
    metadata = deepcopy(document.dataset.metadata)
    if any(
        "unsupported" in w.lower() or "cannot read" in w.lower() for w in scene.warnings
    ):
        raise ValueError(
            "Some source annotations could not be read; export cannot safely discard them: "
            + "; ".join(scene.warnings)
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".fish-edit-", dir=target.parent
    ) as temporary:
        staging = Path(temporary)
        _copy_image(source, staging / target.name, lambda e: None, None)
        (staging / target.with_suffix(".imx").name).write_bytes(
            editable_scene_xml(target.name, scene, metadata)
        )
        write_excel(staging / target.with_suffix(".xlsx").name, report)
        (staging / target.with_suffix(".annotations.json").name).write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        loaded = load_dataset(
            LibraryEntry(
                staging / target.name, staging / target.with_suffix(".imx").name
            )
        )
        if len(loaded.scene.surfaces) != len(scene.surfaces) or sum(
            len(m.points) for m in loaded.scene.measurements
        ) != sum(len(m.points) for m in scene.measurements):
            raise ValueError("Edited annotations failed export reload validation.")
        for expected, actual in zip(scene.surfaces, loaded.scene.surfaces):
            if not np.allclose(
                expected.vertices, actual.vertices, rtol=1e-7, atol=1e-7
            ):
                raise ValueError("Contour geometry changed during export.")
        _publish(staging, destinations)
    return str(target)
