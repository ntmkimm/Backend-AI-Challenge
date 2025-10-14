# BUCCI_GANG_SYSTEM
<p align="center">
  <img src="demo/pipeline.png" alt="System Architecture">
</p>

This tool matches furniture objects to scenes based on text descriptions and context image using a combination of CLIP and InternVL models. It processes scenes with masked areas and finds the most suitable objects to place in those areas based on visual and textual similarity.

## Features

- CLIP & BEiT3 retrieval: Use OpenCLIP and BEiT3 models to find the most likely objects based on text descriptions.

- OCR & ASR: Extract on-screen text and spoken words from video keyframes using their timestamps.

- Object detection: Apply D-FINE and Co-DeTR to detect common (COCO) objects in keyframes.

- Weighted fusion: Combine and re-rank detection results by balancing feature scores within each keyframe.

- Search algorithm: Use a temporal model and dynamic programming to re-rank all keyframes across time for better overall accuracy.

## Demo
<p align="center">
  <img src="demo/demo.png" alt="Demo on web">
</p>

