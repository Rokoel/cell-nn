"""Read-only Imaris image access, including the legacy TIFF container."""
from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
import os
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import tifffile

from .models import Channel, Dataset, LibraryEntry, Metadata, Scene
from .scenes import parse_scene, read_scene, resolve_reference, scene_reference


def decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace').rstrip('\x00')
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'S':
            return decode(value.tobytes())
        if value.dtype.kind == 'U':
            return ''.join(value.ravel()).rstrip('\x00')
        if value.size == 1:
            return str(value.item())
    return str(value)


def attributes(group) -> dict[str, str]:
    return {k: decode(v) for k, v in group.attrs.items()}


def _ids(group, prefix) -> list[int]:
    return sorted(int(k[len(prefix):]) for k in group if k.startswith(prefix) and k[len(prefix):].isdigit())


def _channel(info: dict[str, str], index: int, dtype) -> Channel:
    fallback = [(0, 1, 0), (1, 0, 0), (0.55, 0.25, 1), (0, 0.8, 1), (1, 0.8, 0)]
    color = np.fromstring(info.get('Color', ''), sep=' ')
    limits = np.fromstring(info.get('ColorRange', ''), sep=' ')
    maximum = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
    low, high = tuple(limits) if len(limits) == 2 and limits[1] > limits[0] else (0.0, maximum)
    return Channel(info.get('Name', f'Channel {index + 1}'),
        tuple(np.clip(color[:3], 0, 1)) if len(color) >= 3 else fallback[index % len(fallback)],
        float(low), float(high), max(0.05, float(info.get('GammaCorrection', '1'))),
        float(np.clip(float(info.get('ColorOpacity', '1')), 0, 1)))


def _geometry(info: dict[str, str], shape: tuple[int, int, int]):
    origin = tuple(float(info.get(f'ExtMin{i}', 0)) for i in range(3))
    extent = tuple(float(info.get(f'ExtMax{i}', origin[i] + shape[2-i])) - origin[i] for i in range(3))
    spacing = tuple(e / n for e, n in zip(extent, shape[::-1]))
    if any(s <= 0 or not np.isfinite(s) for s in spacing):
        raise ValueError('Invalid image physical extents; cannot align annotations reliably.')
    unit = info.get('Unit', 'voxel')
    return origin, spacing, 'µm' if unit in ('um', 'micrometer') else unit


class ImarisReader:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f'Image file is missing: {self.path}')
        self.hdf = h5py.is_hdf5(self.path)
        self.metadata = self._hdf_metadata() if self.hdf else self._tiff_metadata()

    def _hdf_metadata(self) -> Metadata:
        with h5py.File(self.path, 'r') as f:
            self.dataset_name = decode(f.attrs.get('DataSetDirectoryName', 'DataSet'))
            self.info_name = decode(f.attrs.get('DataSetInfoDirectoryName', 'DataSetInfo'))
            if self.dataset_name not in f:
                raise ValueError('This HDF5 file does not contain an Imaris DataSet.')
            data = f[self.dataset_name]
            self.channel_ids = []
            levels = {}
            for level in _ids(data, 'ResolutionLevel '):
                group = data[f'ResolutionLevel {level}']
                times = _ids(group, 'TimePoint ')
                if not times:
                    continue
                channels = _ids(group[f'TimePoint {times[0]}'], 'Channel ')
                g = group[f'TimePoint {times[0]}/Channel {channels[0]}']
                attrs = attributes(g)
                shape = tuple(int(attrs.get(f'ImageSize{axis}', g['Data'].shape[i])) for i, axis in enumerate('ZYX'))
                if any(n < 1 or n > actual for n, actual in zip(shape, g['Data'].shape)):
                    raise ValueError(f'Invalid image dimensions at resolution {level}.')
                levels[level] = shape
                if not self.channel_ids:
                    self.channel_ids = channels
                    first_times = times
                    dtype = g['Data'].dtype
            if 0 not in levels:
                raise ValueError('No full-resolution image data was found.')
            info = attributes(f[f'{self.info_name}/Image'])
            origin, spacing, unit = _geometry(info, levels[0])
            channels = [_channel(attributes(f[f'{self.info_name}/Channel {c}'])
                if f'{self.info_name}/Channel {c}' in f else {}, i, dtype)
                for i, c in enumerate(self.channel_ids)]
            return Metadata(self.path, levels[0], origin, spacing, channels, first_times, levels, unit)

    def _tiff_metadata(self) -> Metadata:
        with self.path.open('rb') as stream:
            signature = stream.read(24)
        if signature[:4] not in (b'II*\x00', b'MM\x00*') or b'ImarisDataSet' not in signature:
            raise ValueError('Unsupported Imaris container. Expected HDF5 IMS or Imaris TIFF.')
        with tifffile.TiffFile(self.path) as f:
            raw = f.pages[0].tags['ImageDescription'].value
            parser = ConfigParser(interpolation=None, strict=False)
            parser.optionxform = str
            parser.read_string(raw.decode('latin-1') if isinstance(raw, bytes) else raw)
            info = dict(parser['Image 0.0.0'])
            shape = tuple(int(info[a]) for a in 'ZYX')
            origin, spacing, unit = _geometry(info, shape)
            count = int(info['NumberOfChannels'])
            times = int(parser.get('TimeInfo', 'DatasetTimePoints', fallback='1'))
            self.tiff_pages = [i for i, p in enumerate(f.pages) if p.shape == shape]
            if len(self.tiff_pages) != count * times:
                raise ValueError('Unsupported legacy TIFF channel/time layout.')
            dtype = f.pages[self.tiff_pages[0]].dtype
            channels = [_channel(dict(parser[f'Channel 0.0.0.{c}']), c, dtype) for c in range(count)]
            return Metadata(self.path, shape, origin, spacing, channels, list(range(times)), {0: shape},
                unit, 'Imaris TIFF (legacy)')

    def read(self, time: int = 0, level: int = 0, progress=None) -> list[np.ndarray]:
        if time not in self.metadata.times or level not in self.metadata.levels:
            raise ValueError('Unavailable time point or resolution level.')
        volumes = []
        if self.hdf:
            with h5py.File(self.path, 'r') as f:
                shape = self.metadata.levels[level]
                for i, c in enumerate(self.channel_ids):
                    if progress:
                        progress(f'Reading channel {i + 1}/{len(self.channel_ids)}…')
                    dataset = f[f'{self.dataset_name}/ResolutionLevel {level}/TimePoint {time}/Channel {c}/Data']
                    volumes.append(dataset[tuple(slice(0, n) for n in shape)])
        else:
            with tifffile.TiffFile(self.path) as f:
                count = len(self.metadata.channels)
                for c in range(count):
                    if progress:
                        progress(f'Reading channel {c + 1}/{count}…')
                    # Legacy TIFF page row zero is the lower Y edge (Orientation=4).
                    volumes.append(f.pages[self.tiff_pages[time * count + c]].asarray())
        return volumes

    def embedded_scene(self) -> Scene:
        if not self.hdf:
            return Scene()
        with h5py.File(self.path, 'r') as f:
            for path in ('Scene/Content', 'Scene/Scene', 'Scene8/Content'):
                if path in f and isinstance(f[path], h5py.Dataset):
                    raw = f[path][()]
                    data = raw.tobytes() if isinstance(raw, np.ndarray) else raw
                    try:
                        return parse_scene(data, self.path)
                    except Exception as exc:
                        return Scene(warnings=[f'Cannot read embedded scene: {exc}'])
            if 'Scene' in f or 'Scene8' in f:
                return Scene(warnings=['This file contains a binary scene schema that is not supported. '
                    'An Imaris 5/6 XML .imx companion can be read.'])
        return Scene()


def entry_for_path(path: str | Path) -> LibraryEntry:
    path = Path(path).resolve()
    if path.suffix.lower() == '.imx':
        reference = scene_reference(path)
        try:
            image = resolve_reference(path, reference)
        except FileNotFoundError:
            # Keep the scene discoverable and report the absent image when opened.
            image = path.with_suffix('.ims')
        return LibraryEntry(image, path)
    if path.suffix.lower() != '.ims':
        raise ValueError('Select an .ims image or .imx scene file.')
    companions = sorted((p for p in path.parent.iterdir() if p.suffix.lower() == '.imx'), key=lambda p: p.name.casefold())
    matching = [p for p in companions if p.stem.casefold() == path.stem.casefold()]
    for candidate in matching + [p for p in companions if p not in matching]:
        try:
            if resolve_reference(candidate, scene_reference(candidate)) == path:
                return LibraryEntry(path, candidate)
        except ET.ParseError:
            if candidate in matching:
                return LibraryEntry(path, candidate)
        except (OSError, ValueError):
            continue
    return LibraryEntry(path)


def scan_library(folder: str | Path) -> tuple[list[LibraryEntry], list[str]]:
    entries = []
    warnings = []
    for parent, _, files in os.walk(folder, onerror=lambda e: warnings.append(str(e))):
        images = {Path(parent, n).resolve() for n in files if n.lower().endswith('.ims')}
        used = set()
        for name in sorted(files, key=str.casefold):
            if name.lower().endswith('.imx'):
                try:
                    entry = entry_for_path(Path(parent, name))
                    entries.append(entry)
                    used.add(entry.image)
                except Exception as exc:
                    warnings.append(f'{name}: {exc}')
        entries.extend(LibraryEntry(p) for p in images - used)
    return sorted(set(entries), key=lambda e: (str(e.image).casefold(), str(e.scene))), warnings


def load_dataset(entry: LibraryEntry, time=0, level=0, progress=None) -> Dataset:
    reader = ImarisReader(entry.image)
    if progress:
        progress('Reading saved annotations…')
    if entry.scene:
        try:
            scene = read_scene(entry.scene)
        except Exception as exc:
            scene = Scene(warnings=[f'Cannot read {entry.scene.name}: {exc}'], source=entry.scene)
    else:
        scene = reader.embedded_scene()
        if not scene.surfaces and not scene.measurements and not scene.warnings:
            scene.warnings.append('No saved annotations found. Keep the .imx companion beside the .ims image.')
    return Dataset(reader.metadata, scene, reader.read(time, level, progress), time, level)
