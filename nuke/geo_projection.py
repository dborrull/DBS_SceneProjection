# DBS_SceneProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""DBS_SceneProjection entry point for Nuke: plate -> camera + basic geometry (planes / boxes).

Install: copy geo_projection.py and the geoproj folder to ~/.nuke (or add this folder to the
plugin path), then in ~/.nuke/menu.py:

    import geo_projection
    geo_projection.install_menu()

Needs a ComfyUI with MoGe weights + comfy_nodes/nuke_geo_projection (and comfyui-rmbg for
SAM3 labels), and Blender 4.2+.
"""

from geoproj import __version__
from geoproj.nuke_ui import (add_frame, cornerpin_card, create_node, extract, install_menu, isolate_classic, run, show_labels,
                             test_connection)

__all__ = ["__version__", "add_frame", "cornerpin_card", "create_node", "extract", "install_menu", "isolate_classic", "run",
           "show_labels", "test_connection"]
