import importlib.metadata
import pathlib

import anywidget
import traitlets

try:
    __version__ = importlib.metadata.version("lloom")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

_DEV = False # switch to False for production

if _DEV:
  # from `npx vite`
  ESM = "http://localhost:5173/src/index.js?anywidget"
  CSS = ""
else:
  # from `npx vite build`
  # Path to static from lloom_ai/src/lloom_ai (the python package)
  bundled_assets_dir = pathlib.Path(__file__).parent / "static"
  ESM = (bundled_assets_dir / "index.js").read_text()
  CSS = (bundled_assets_dir / "style.css").read_text()
  

"""
MATRIX WIDGET
Widget instantiated with anywidget that displays the matrix visualization
"""
class MatrixWidget(anywidget.AnyWidget):
    _esm = ESM
    _css = CSS
    name = traitlets.Unicode().tag(sync=True)

    data = traitlets.Unicode().tag(sync=True)  # syncs the widget's `data` property
    data_items = traitlets.Unicode().tag(sync=True)  # syncs the widget's `data_items` property
    data_items_wide = traitlets.Unicode().tag(sync=True)  # syncs the widget's `data_items_wide` property
    metadata = traitlets.Unicode().tag(sync=True)  # syncs the widget's `metadata` property
