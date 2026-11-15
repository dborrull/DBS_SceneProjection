# DBS_SceneProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Qt widgets for the DBS_SceneProjection panel (PyCustom_Knob).

LabelsEditor: a tall, always open text box for the extra labels. Nuke's own multiline knob
shows as one line with a scroll bar, which does not look like a place to type. The text is
stored in the node's hidden 'extra_labels' string knob, so the rest of the gizmo (and scripts
opened without this module) only ever see that knob."""

import nuke

try:
    from PySide6 import QtWidgets
except ImportError:                          # Nuke 13-15
    from PySide2 import QtWidgets

PLACEHOLDER = ("one label per line:  text:role   or   text:object:shape\n"
               "e.g.   lamp:object:revolve\n"
               "       rug:floor\n"
               "       painting:ignore\n"
               "roles: floor wall ceiling object backdrop ignore - shapes: box card cylinder cone sphere revolve relief")


class LabelsEditor(object):
    def __init__(self):
        self.node = nuke.thisNode()
        self.edit = None

    def makeUI(self):
        self.edit = QtWidgets.QPlainTextEdit()
        self.edit.setPlaceholderText(PLACEHOLDER)
        self.edit.setMinimumHeight(110)
        self.edit.setPlainText(self.node["extra_labels"].value())
        self.edit.textChanged.connect(self._changed)
        return self.edit

    def _changed(self):
        text = self.edit.toPlainText()
        if text != self.node["extra_labels"].value():
            self.node["extra_labels"].setValue(text)

    def updateValue(self):
        """Called by Nuke when the knob changes from outside (undo, script load)."""
        if self.edit is not None and self.edit.toPlainText() != self.node["extra_labels"].value():
            self.edit.setPlainText(self.node["extra_labels"].value())
