"""Human review of candidate contours, stored separately from source annotations."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PySide6 import QtCore,QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from ..pipeline.runner import _json


class CandidateReview:
    def __init__(self,predictions):
        self.path=Path(predictions)
        with np.load(self.path,allow_pickle=False) as archive:
            self.labels=archive['labels'];self.metadata=json.loads(str(archive['metadata']))
        self.review_path=self.path.with_suffix('.review.json')
        digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.review=dict(version=1,predictions=str(self.path.resolve()),prediction_sha256=digest,
            image=self.metadata['image'],decisions={})
        if self.review_path.exists():
            saved=json.loads(self.review_path.read_text())
            if saved['prediction_sha256']!=digest:raise ValueError('Saved review belongs to different predictions.')
            self.review=saved
        self.ids=[int(n) for n in np.unique(self.labels) if n]
        self.boxes=ndi.find_objects(self.labels)

    def decide(self,label,decision):
        if label not in self.ids or decision not in ('accepted','rejected','unsure'):raise ValueError('Invalid review decision.')
        self.review['decisions'][str(label)]=dict(decision=decision,updated=datetime.now(timezone.utc).isoformat())
        _json(self.review_path,self.review)

    def export_confirmed(self):
        ids=[int(k) for k,v in self.review['decisions'].items() if v['decision']=='accepted']
        labels=np.where(np.isin(self.labels,ids),self.labels,0).astype(self.labels.dtype)
        metadata=dict(self.metadata,cells=[c for c in self.metadata['cells'] if c['label'] in ids],
            review_file=str(self.review_path.resolve()),selection='Only human-confirmed correct contours')
        # Old evaluation scores describe the original output, not this subset.
        metadata.pop('metrics',None)
        output=self.path.with_suffix('.reviewed.npz')
        np.savez_compressed(output,labels=labels,metadata=np.array(json.dumps(metadata,allow_nan=False)))
        return output


class ReviewDialog(QtWidgets.QDialog):
    def __init__(self,predictions,image_path,parent=None,on_export=None):
        super().__init__(parent)
        self.on_export=on_export
        self.data=CandidateReview(predictions)
        if not self.data.ids:raise ValueError('This image has no accepted candidates to review.')
        self.image=np.load(image_path,mmap_mode='r')
        if self.image.shape!=self.data.labels.shape:raise ValueError('Cached image and prediction grids differ.')
        self.index=0
        self.setWindowTitle('Review cell contours');self.resize(1100,680)
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('Confirm only when both the cell and its contour are correct. Inspect several slices; use Unsure when evidence is incomplete. '
            'Decisions save automatically. Review labels are not converted into background masks.')
        note.setWordWrap(True);layout.addWidget(note)
        self.title=QtWidgets.QLabel();layout.addWidget(self.title)
        self.figure=Figure(figsize=(10,4),layout='constrained');self.canvas=FigureCanvasQTAgg(self.figure)
        self.axes=self.figure.subplots(1,3);layout.addWidget(self.canvas,1)
        row=QtWidgets.QHBoxLayout();self.sliders=[]
        for axis,name in enumerate(('XY: Z slice','XZ: Y slice','YZ: X slice')):
            column=QtWidgets.QVBoxLayout();column.addWidget(QtWidgets.QLabel(name))
            slider=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal);slider.valueChanged.connect(self.draw)
            self.sliders.append(slider);column.addWidget(slider);row.addLayout(column)
        layout.addLayout(row)
        row=QtWidgets.QHBoxLayout()
        for text,decision in [('Correct cell and contour','accepted'),('False / incorrect contour','rejected'),('Unsure','unsure')]:
            button=QtWidgets.QPushButton(text);button.clicked.connect(lambda checked=False,d=decision:self.decide(d));row.addWidget(button)
        layout.addLayout(row)
        row=QtWidgets.QHBoxLayout()
        for text,offset in [('Previous',-1),('Next',1)]:
            button=QtWidgets.QPushButton(text);button.clicked.connect(lambda checked=False,o=offset:self.show_candidate(o));row.addWidget(button)
        export=QtWidgets.QPushButton('Export confirmed contours');export.clicked.connect(self.export);row.addWidget(export)
        layout.addLayout(row)
        self.summary=QtWidgets.QLabel();self.summary.setWordWrap(True);layout.addWidget(self.summary)
        self.show_candidate(0)

    def show_candidate(self,offset):
        self.index=max(0,min(len(self.data.ids)-1,self.index+offset))
        label=self.data.ids[self.index];box=self.data.boxes[label-1]
        spacing=np.asarray(self.data.metadata['spacing_zyx']);margin=np.ceil(2/spacing).astype(int)
        self.crop=tuple(slice(max(0,s.start-int(m)),min(n,s.stop+int(m))) for s,m,n in zip(box,margin,self.image.shape))
        for slider,s in zip(self.sliders,self.crop):
            slider.blockSignals(True);slider.setRange(s.start,s.stop-1);slider.setValue((s.start+s.stop-1)//2);slider.blockSignals(False)
        self.draw()

    def draw(self):
        label=self.data.ids[self.index];image=self.image[self.crop];mask=self.data.labels[self.crop]==label
        spacing=np.asarray(self.data.metadata['spacing_zyx'])
        for axis,(ax,slider) in enumerate(zip(self.axes,self.sliders)):
            ax.clear();index=slider.value()-self.crop[axis].start
            plane=np.take(image,index,axis=axis);border=np.take(mask,index,axis=axis)
            remaining=[i for i in range(3) if i!=axis]
            ax.imshow(plane,cmap='gray',vmin=0,vmax=1,aspect=spacing[remaining[0]]/spacing[remaining[1]])
            if border.any() and not border.all():ax.contour(border,levels=[.5],colors=['#00d590'],linewidths=1)
            ax.set_title(f'{("XY","XZ","YZ")[axis]} · slice {slider.value()}');ax.set_xticks([]);ax.set_yticks([])
        decision=self.data.review['decisions'].get(str(label),{}).get('decision','unreviewed')
        self.title.setText(f'{Path(self.data.metadata["image"]).name} · candidate {self.index+1} / {len(self.data.ids)} · label {label} · {decision}')
        values=[v['decision'] for v in self.data.review['decisions'].values()]
        self.summary.setText(f'Confirmed {values.count("accepted")} · rejected {values.count("rejected")} · unsure {values.count("unsure")} · '
            f'unreviewed {len(self.data.ids)-len(values)}. These selected reviews do not establish dataset-wide precision.')
        self.canvas.draw_idle()

    def decide(self,decision):
        self.data.decide(self.data.ids[self.index],decision);self.show_candidate(1)

    def export(self):
        path=self.data.export_confirmed()
        self.summary.setText(f'Confirmed contours saved to {path}.')
        if self.on_export:self.on_export(path)
