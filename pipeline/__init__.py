"""
pipeline
========

Wound-closure analysis pipeline, structured to match the Team-5 workflow diagram.

    PHASE 1  Pre-processing & Segmentation
        original tiff  ->  segment (Cellpose)  ->  mask.tiff
                       ->  mask cleaning        ->  cleaned_mask.tiff

    PHASE 2  Three analytical branches
        Step 3  Cell Tracking      -> tracked_mask.tiff   (track.py)
        Step 4  Morphology         -> morphology.csv      (morphology.py)
        Step 5  Edge Kinematics    -> edge_velocity.csv   (kinematics.py)

Every stage reads/writes plain ``(T, Y, X)`` arrays / labelled TIFFs, is driven
by :class:`pipeline.config.Params`, and is orchestrated by
:func:`pipeline.run.run_pipeline`.  A Streamlit GUI (``app.py``) exposes the
parameters and runs it.
"""

from . import (config, segment, clean, track, morphology, kinematics,
               viz, intensity, run, boundary)

__all__ = ["config", "segment", "clean", "track", "morphology", "kinematics",
           "viz", "intensity", "run", "boundary"]
