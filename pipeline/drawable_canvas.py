"""Reliable wrapper around streamlit-drawable-canvas.

The upstream 0.9.3 frontend always prepends the Streamlit origin to a
background URL. That breaks embedded ``data:`` images and can leave a blank
canvas when Streamlit's temporary media URL expires. This module installs a
versioned, locally patched copy of the component frontend and sends the
background as an embedded PNG.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components
from PIL import Image


_COMPONENT_VERSION = "wound_canvas_v2"
_OLD_SOURCE = "e.src=n+h"
_NEW_SOURCE = 'e.src=h&&h.startsWith("data:")?h:n+h'


def _patched_build_dir():
    import streamlit_drawable_canvas

    source = (Path(streamlit_drawable_canvas.__file__).resolve().parent /
              "frontend" / "build")
    if not source.is_dir():
        raise RuntimeError(f"Drawing-canvas frontend not found at {source}")

    target = Path(tempfile.gettempdir()) / _COMPONENT_VERSION
    marker = target / ".patched"
    if marker.exists():
        return str(target)

    staging = target.with_name(target.name + "_staging")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source, staging)

    bundle = staging / "static" / "js" / "main.80185090.chunk.js"
    source_text = bundle.read_text(encoding="utf-8")
    if _OLD_SOURCE not in source_text and _NEW_SOURCE not in source_text:
        raise RuntimeError("Unsupported drawing-canvas frontend bundle.")
    bundle.write_text(source_text.replace(_OLD_SOURCE, _NEW_SOURCE),
                      encoding="utf-8")
    marker_staging = staging / ".patched"
    marker_staging.write_text(_COMPONENT_VERSION, encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)
    os.replace(staging, target)
    return str(target)


_component_func = components.declare_component(
    _COMPONENT_VERSION, path=_patched_build_dir())


@dataclass
class CanvasResult:
    image_data: np.ndarray | None = None
    json_data: dict | None = None


def _data_url_to_image(data_url):
    _, encoded = data_url.split(";base64,", 1)
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def _background_data_url(image, height, width):
    resampling = getattr(Image, "Resampling", Image)
    image = image.convert("RGB").resize((width, height), resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(
        buf.getvalue()).decode("ascii")


def stable_drawable_canvas(
    *, background_image, fill_color="#eee", stroke_width=20,
    stroke_color="black", update_streamlit=True, height=400, width=600,
    drawing_mode="freedraw", initial_drawing=None, display_toolbar=True,
    key=None,
):
    """Render a drawable canvas whose background cannot expire between reruns."""
    drawing = ({"version": "4.4.0"} if initial_drawing is None
               else dict(initial_drawing))
    drawing["background"] = ""
    component_value = _component_func(
        fillColor=fill_color,
        strokeWidth=stroke_width,
        strokeColor=stroke_color,
        backgroundColor="",
        backgroundImageURL=_background_data_url(
            background_image, int(height), int(width)),
        realtimeUpdateStreamlit=bool(update_streamlit and
                                     drawing_mode != "polygon"),
        canvasHeight=int(height),
        canvasWidth=int(width),
        drawingMode=drawing_mode,
        initialDrawing=drawing,
        displayToolbar=display_toolbar,
        displayRadius=3,
        key=key,
        default=None,
    )
    if component_value is None:
        return CanvasResult()
    return CanvasResult(
        np.asarray(_data_url_to_image(component_value["data"])),
        component_value["raw"],
    )
