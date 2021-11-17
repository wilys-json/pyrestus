# Data Processing Package

Functions:

- [Data Randomization](## Data Randomization)

## Requirements
---
1. Python 3.x
2. Python libraries:

- `numpy`
- `pandas`-
- `opencv-python` (OpenCV)
- `tqdm`


## Installing Libraries & Dependencies
---
### MacOS

1. Clone this repository: `git clone https://github.com/wlamuchk/DataProcessing`

2. Open Terminal, navigate to the program directory

3. Type `/bin/bash setup.sh` - this will set up everything you need for this program


### Windows

1. Clone this repository: `git clone https://github.com/wlamuchk/DataProcessing`

2. Dowload and install [Python3](#https://www.python.org/ftp/python/3.8.9/python-3.8.9-amd64.exe)

3. Open Command Prompt

4. Type `python -m pip install -r requirements.txt`


## Data Randomization
---

This module allows you to randomize video and image data.

### Basic Usage
---

You can randomize your AVI dataset by:

*Note: change `python3` to `python` on Windows.*

1. Open Terminal, go to the program directory

2. Type `python3  randomize`

3. Drag and drop the folder containing the AVI files, such that the command becomes `python3 randomize $path/to/folder`,
where `$path/to/folder` is the path of the target

When the program finishes, you shall find the randomized files in the `output/` folder

### Unhiding the encoding file
---

When the program randomizes the files and generate the shuffled & duplicated files by encoding each file with a number, it also generates a hidden matching list of the encoded file name with the respective original file called `Randomized.csv`. You can unhide this file by:

1. Open Terminal, go to the program directory

2. Type `python3 randomize -hide`

You shall now see a file called `Randomized.csv`

To hide this file again, do: `python3 randomize +hide`


### Advanced Usage
---

#### Customizing the percentage of repeated(duplicated) files

This program allows you to decide how much of the origin data is repeated so that you can examine the intra-rater reliability. To do so, add `--duplicate-factor $floating/point/number` to the command:

e.g. `python3 randomize $path/to/data/folder --duplicate-factor 0.5` will shuffle again your data and repeat 50% of the data in the generated set.

Make sure you input a floating pointing number of : 0 < number <= 1

Default value: 0.3

#### Customizing the output path

You can change the directory of the generate files by adding `--output-folder $path/to/output`.
For example, `python3 randomize $path/to/data/folder --output-folder ~/generated` will place the generated files under a folder called `generated` under your root directory.
