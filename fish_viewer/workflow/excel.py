"""Excel exports with numeric volumes, signal centers, B points and distances."""
from collections import Counter
from pathlib import Path
import os
import tempfile

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill


def write_excel(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    workbook.remove(workbook.active)
    volumes = {r['cell_id']: r for r in report.get('cell_volumes', [])}
    totals = Counter(p['cell_id'] for p in report['points'])
    refinement = report.get('boundary_refinement', {})
    refinements = {r['cell_id']: r for r in refinement.get('cells', [])}
    def sheet(name, headers, rows):
        page = workbook.create_sheet(name)
        page.append(headers)
        for row in rows:
            # Text from input filenames/channel names must never become formulas.
            page.append(row)
            for cell in page[page.max_row]:
                if isinstance(cell.value, str):
                    cell.data_type = 's'
        page.freeze_panes = 'A2'
        page.auto_filter.ref = page.dimensions
        for cell in page[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='205E58')
        for col in page.columns:
            page.column_dimensions[col[0].column_letter].width = min(60, max(16, len(str(col[0].value))+2))
    extra_headers = ['Boundary refinement', 'Refinement fallback reason'] if refinement else []
    sheet('Cells', ['Cell', 'Volume (um^3)', 'Voxel volume (um^3)', 'Signals (all channels)', 'Volume status', 'Cell at image edge']+extra_headers,
          [[cell, volumes.get(cell, {}).get('volume_um3'), volumes.get(cell, {}).get('voxel_volume_um3'), totals[cell],
            volumes.get(cell, {}).get('volume_status', 'unavailable'), volumes.get(cell, {}).get('cell_at_image_edge')]
            + ([('crop contour' if report.get('cell_detector',{}).get('engine')=='cell-center-box3d' else 'refined')
                if refinements.get(cell, {}).get('accepted') else 'detector contour retained',
                refinements.get(cell, {}).get('fallback_reason')] if refinement else [])
           for cell in sorted(set(volumes) | set(totals))])
    sheet('Signals', ['Signal', 'Cell', 'Channel (1-based)', 'Channel name', 'Score', 'Cell volume (um^3)',
                     'A x (um)', 'A y (um)', 'A z (um)', 'B x (um)', 'B y (um)', 'B z (um)', 'Distance (um)', 'Border at image edge'],
          [[p['signal_id'], p['cell_id'], p['channel']+1, report['channels'][p['channel']]['name'], p.get('score'),
            volumes.get(p['cell_id'], {}).get('volume_um3'), *p['xyz'], *p.get('border_xyz', [None]*3),
            p.get('distance_um'), p.get('border_at_image_edge')] for p in report['points']])
    sheet('Channel counts', ['Cell', 'Channel (1-based)', 'Channel name', 'Signals', 'Cell volume (um^3)'],
          [[r['cell_id'], r['channel']+1, r['channel_name'], r['count'], volumes.get(r['cell_id'], {}).get('volume_um3')]
           for r in report['counts']])
    sheet('Summary', ['Property', 'Value'], [
        ['Image', str(report.get('image', ''))], ['Time point (1-based)', report.get('time', 0)+1],
        ['Signal model', str(report.get('model_directory', ''))], ['Cells', len(set(totals) | set(volumes))],
        ['Signals', len(report['points'])], ['Volume definition', report.get('volume_method', 'unavailable')],
        ['Distance definition', report.get('measurement', {}).get('method', 'not measured')],
        ['Boundary model', str(refinement.get('model', 'none'))],
        ['Cell detector', str(report.get('cell_detector', {}).get('engine', 'StarDist'))],
        ['Cell retention', 'At least one accepted signal in any non-membrane channel'],
        ['Units', 'Coordinates and distances: um; volumes: um^3']])
    fd, temporary = tempfile.mkstemp(suffix='.xlsx', prefix='.fish-', dir=path.parent)
    os.close(fd)
    try:
        workbook.save(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
        workbook.close()
    return path
