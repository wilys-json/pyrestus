# Segmentation Extraction & Evaluation

This module offers basic functionalities for extracting raw segmentations performed in common photo editing softwares in the command line interface (CLI) - for annotations resulting in other formats like .xml from CVAT, please use the `annotation` module (coming soon).


## Extraction of Raw Segmentation
  To extract segmentations, open the CLI (e.g. Terminal in MacOS) and type:

  `python3 compareimages path/to/segmentations --generate-mask`

#### Optional flags

`-r`, `--show-original` - the original segmentation will be shown during extraction (default: false)

`-b`, `--show-binary` - the extraction will be shown as a binary image (default: false)

`-l`, `--lines-only` - only extract contours / lines from segmentation; otherwise, a flood fill method will be applied to the segmentation (default: false)

`-m`, `--show-mask` - show the final extraction mask (default:false)

`-o`, `--output` `path/to/output/directory` - the output directory (default: `outputs`)

## Evaluation Metrics
```math
SE = \frac{\sigma}{\sqrt{n}}
```
