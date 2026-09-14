from pathlib import Path

import numpy as np
import pytest

from fish_viewer.models import composite_slice
from fish_viewer.reader import ImarisReader, entry_for_path, load_dataset, scan_library
from fish_viewer.scenes import parse_scene


def test_hdf_valid_extent_and_time_channels(ims_file):
    reader = ImarisReader(ims_file)
    assert reader.metadata.shape == (2, 3, 4)
    assert reader.metadata.origin == (10, 20, 30)
    assert reader.metadata.spacing == (2, 2, 2)
    assert reader.metadata.center_origin == (11, 21, 31)
    assert reader.metadata.unit == 'µm'
    assert reader.metadata.times == [0, 1]
    channels = reader.read(time=1)
    assert channels[0].shape == (2, 3, 4)
    assert channels[0].dtype == np.uint16
    assert np.all(channels[0] == 30)
    assert np.all(channels[1] == 35)


def test_resolution_retains_physical_extent(ims_file):
    data = load_dataset(entry_for_path(ims_file), level=1)
    assert data.spacing == (4, 3, 2)
    assert data.center_origin == (12, 21.5, 31)
    assert data.metadata.extent == (8, 6, 4)
    with pytest.raises(ValueError):
        ImarisReader(ims_file).read(time=8)


def test_composite_channels_and_source_immutability(ims_file):
    data = load_dataset(entry_for_path(ims_file))
    originals = [v.copy() for v in data.volumes]
    rgb = composite_slice(data, 0, 0)
    assert tuple(rgb[0, 0]) == (38, 25, 0)
    data.metadata.channels[0].visible = False
    rgb = composite_slice(data, 0, 0)
    assert tuple(rgb[0, 0]) == (38, 0, 0)
    data.metadata.channels[1].brightness = 0.2
    assert composite_slice(data, 0, 0)[0, 0, 0] > rgb[0, 0, 0]
    for before, after in zip(originals, data.volumes):
        np.testing.assert_array_equal(before, after)
    assert ImarisReader(ims_file).read()[0][0, 0, 0] == 10


def write_scene(path, reference):
    path.write_text(f'<bpImarisApplication><bpCommandLoadDataSet file="{reference}"/></bpImarisApplication>')


def test_recursive_library_pairs_relocated_windows_reference(tmp_path):
    folder = tmp_path / 'deep' / 'nested'
    folder.mkdir(parents=True)
    image = folder / 'Sample.IMS'
    image.touch()
    scene = folder / 'alternative.imx'
    write_scene(scene, r'C:\old\location\Sample.IMS')
    entries, errors = scan_library(tmp_path)
    assert not errors
    assert len(entries) == 1
    assert entries[0].image == image.resolve()
    assert entries[0].scene == scene.resolve()
    assert entry_for_path(image) == entries[0]


def test_scene_reference_takes_priority_over_filename(tmp_path):
    image_a = tmp_path / 'a.ims'
    image_b = tmp_path / 'b.ims'
    image_a.touch()
    image_b.touch()
    scene = tmp_path / 'a.imx'
    write_scene(scene, 'b.ims')
    assert entry_for_path(image_a).scene is None
    assert entry_for_path(scene).image == image_b
    assert len(scan_library(tmp_path)[0]) == 2


def test_missing_images_are_kept_and_fail_clearly(tmp_path):
    scene = tmp_path / 'missing.imx'
    write_scene(scene, 'missing.ims')
    entry = scan_library(tmp_path)[0][0]
    with pytest.raises(FileNotFoundError, match='missing'):
        load_dataset(entry)


def test_scene_blocks_join_before_triangle_indexing():
    scene = parse_scene('''<bpImarisApplication><bpSurfacesViewer>
      <bpSurfaceComponent><name>Cell 1</name></bpSurfaceComponent>
      <bpSurfaces><surface>
        <vertices>"0 0 0 0 0 1 1 0 0 0 0 1"</vertices>
        <vertices>"0 1 0 0 0 1"</vertices>
        <triangles>"0 1 2"</triangles><timeIndex>1</timeIndex>
      </surface></bpSurfaces></bpSurfacesViewer></bpImarisApplication>''')
    assert not scene.warnings
    assert scene.surfaces[0].time == 1
    np.testing.assert_array_equal(scene.surfaces[0].vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(scene.surfaces[0].triangles, [[0, 1, 2]])


def test_broken_surface_is_reported_without_losing_measurements():
    scene = parse_scene('''<bpImarisApplication>
    <bpSurfacesViewer><bpSurfaceComponent><name>Broken cell</name></bpSurfaceComponent>
      <bpSurfaces><surface><vertices>0 0 0 0 0 1</vertices><triangles>0 1 2</triangles></surface></bpSurfaces>
    </bpSurfacesViewer>
    <bpMeasurementPoints><Point Name="A" Position="1 2 3"/><Point Name="B" Position="4 6 3"/></bpMeasurementPoints>
    </bpImarisApplication>''')
    assert not scene.surfaces
    assert 'Broken cell' in scene.warnings[0]
    assert scene.measurements[0].length == 5
    assert scene.measurements[0].labels == ['A', 'B']


def test_legacy_pairwise_connections_stay_within_groups():
    points = ''.join(f'<bpMeasurementPoint><bpSurfaceComponent><name>P{i}</name></bpSurfaceComponent><Position x="{i}" y="0" z="0"/></bpMeasurementPoint>' for i in range(4))
    scene = parse_scene(f'<root><bpComponentGroup><bpSurfaceComponent><name>Pairs</name>{points}</bpSurfaceComponent><LineConnection value="Pairwise"/></bpComponentGroup></root>')
    assert scene.measurements[0].edges == [(0, 1), (2, 3)]
    assert scene.measurements[0].length == 2


def test_unmeshed_contour_uses_calibrated_plane(ims_file):
    scene = parse_scene('''<root><bpManualSurface><direction>2</direction><store>
      <plane/><plane><line><vertex x="11" y="21"/><vertex x="13" y="21"/><vertex x="12" y="23"/></line></plane>
    </store></bpManualSurface></root>''')
    assert len(scene.contours) == 1
    vertices = scene.contours[0].vertices(ImarisReader(ims_file).metadata)
    np.testing.assert_array_equal(vertices[:, 2], [33, 33, 33])


@pytest.mark.integration
def test_reference_annotations_and_voxel_data(reference):
    entry = entry_for_path(reference)
    data = load_dataset(entry)
    assert data.metadata.shape == (42, 512, 512)
    assert len(data.metadata.channels) == 3
    assert len(data.scene.surfaces) == 10
    assert len(data.scene.measurements) == 11
    assert not data.scene.warnings
    for mesh in data.scene.surfaces:
        assert mesh.triangles.max() < len(mesh.vertices)
        assert np.all(mesh.vertices >= np.asarray(data.metadata.origin) - 0.5)
        assert np.all(mesh.vertices <= np.asarray(data.metadata.origin) + data.metadata.extent + np.asarray([0.5, 0.5, 0.5]))
    assert np.count_nonzero(data.volumes[0]) > 100


@pytest.mark.integration
def test_legacy_imaris_tiff_pixels_and_annotations(reference):
    path = reference.parents[1] / 'FISH antiLAD/18327antiLAD2.ims'
    data = load_dataset(entry_for_path(path))
    assert data.metadata.format == 'Imaris TIFF (legacy)'
    assert data.metadata.unit == 'µm'
    assert len(data.volumes) == 2
    assert data.scene.surfaces
    assert len(data.scene.measurements) == 10
    assert sum(len(m.edges) for m in data.scene.measurements) == 7
    assert np.count_nonzero(data.volumes[0]) > 100
    assert not data.scene.warnings
