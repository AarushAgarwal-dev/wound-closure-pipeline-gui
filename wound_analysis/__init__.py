"""
wound_analysis
==============

Custom image-analysis toolkit for zebrafish tailfin wound-closure imaging,
built for the EMBRIO Design Challenge (Team 5).  It runs on the real exported
confocal time-lapse in ``Wound/`` (membrane / 561 channel).

Three objectives, mirroring the challenge description:

  1. Edge velocity / wound-closure speed (symmetric vs. asymmetric;
     constant vs. time-varying).  Action plan:
     edge detection -> edge sampling -> define edge window ->
     quantify intensity -> edge point tracking (distance) ->
     velocity calculation -> intensity/velocity correlation -> plotting.

  2. Cell-shape characterization through wound closure (segmentation +
     shape metrics: area, circularity, aspect ratio, elongation).

  3. Cell counting & intercalation: number of cells adjacent to the wound
     edge over time, neighbour layers (1st/2nd/3rd), and intercalation
     (neighbour-exchange) event detection.

Modules
-------
io_utils      load the tif stack + pixel/time calibration
detection     tissue mask + per-frame wound (dark hole) detection
edge_velocity Objective 1
segmentation  Objective 2
intercalation Objective 3
plotting      shared figure helpers
"""

__version__ = "1.0.0"
