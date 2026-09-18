"""Scrollable panels that keep plots usable on smaller desktops."""
from PySide6 import QtCore, QtWidgets


class PanelScrollArea(QtWidgets.QScrollArea):
    def __init__(self, panel):
        super().__init__()
        # Explicit sizing avoids Qt's recursive height-for-width negotiation
        # between the scroll area, word-wrapped labels and Matplotlib canvases.
        self.setWidgetResizable(False)
        self.setWidget(panel)
        self.setMinimumSize(320, 180)
        self.resize_content()

    def resize_content(self):
        panel = self.widget()
        if panel:
            panel.resize(self.viewport().size().expandedTo(panel.minimumSizeHint()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_content()

    def eventFilter(self, watched, event):
        if watched is self.widget() and event.type() == QtCore.QEvent.Type.LayoutRequest:
            self.resize_content()
        return super().eventFilter(watched, event)
