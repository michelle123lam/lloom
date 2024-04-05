import importlib.metadata
import pathlib

import anywidget
import traitlets

import nltk
nltk.download('punkt')

try:
    __version__ = importlib.metadata.version("lloom")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

_DEV = False # switch to False for production

if _DEV:
  # from `npx vite`
  ESM = "http://localhost:5173/src/index.js?anywidget"
  ESM_select = "http://localhost:5173/src/index_select.js?anywidget"
  CSS = ""
  CSS_select = ""
else:
  # from `npm run build`
  # Path to static from text_lloom/src/text_lloom (the python package)
  bundled_assets_dir = pathlib.Path(__file__).parent / "static"
  ESM = (bundled_assets_dir / "index.js").read_text()
  CSS = (bundled_assets_dir / "index.css").read_text()
  ESM_select = (bundled_assets_dir / "index_select.js").read_text()
  CSS_select = (bundled_assets_dir / "index_select.css").read_text()
  

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
    slice_col = traitlets.Unicode().tag(sync=True)  # syncs the widget's `slice_col` property
    norm_by = traitlets.Unicode().tag(sync=True)  # syncs the widget's `norm_by` property
  
"""
CONCEPT SELECT WIDGET
Widget instantiated with anywidget that displays the concepts for selection
"""
class ConceptSelectWidget(anywidget.AnyWidget):
    _esm = ESM_select
    _css = CSS_select
    name = traitlets.Unicode().tag(sync=True)

    data = traitlets.Unicode().tag(sync=True)  # syncs the widget's `data` property
