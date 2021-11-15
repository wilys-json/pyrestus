# Randomizing AVI Data for Intra- and Inter-rater Reliability

A mini Python program to randomize AVI data.

## Requirements

1. Python 3.x
2. Python libraries: `numpy`, `pandas`, `opencv-python`

## Installing Libraries & Dependencies

1. Open Terminal, go to the program directory

2. Type `/bin/bash setup.sh` - this will set up everything you need for this program


## Usage

### Basic Usage

You can randomize your AVI dataset by:

1. Open Terminal, go to the program directory

2. Type `python3 main.py `

3. Drag and drop the folder containing the AVI files, such that the command becomes `python3 main.py $path/to/folder`,
where `$path/to/folder` is the path of the target

When the program finishes, you shall find the randomized files in the `output/` folder

### Unhiding the encoding file

When the program randomizes the files and generate the shuffled & duplicated files by encoding each file with a number, it also generates a hidden matching list of the encoded file name with the respective original file called `Randomized.csv`. You can unhide this file by:

1. Open Terminal, go to the program directory

2. Type `python3 main.py -hide`

You shall now see a file called `Randomized.csv`

To hide this file again, do: `python3 main.py +hide`


### Advanced Usage

#### Customizing the percentage of repeated(duplicated) files

This program allows you to decide how much of the origin data is repeated so that you can examine the intra-rater reliability. To do so, add `--duplicate-factor $floating/point/number` to the command:

e.g. `python3 main.py --duplicate-factor 0.5` will shuffle again your data and repeat 50% of the data in the generated set.

Make sure you input a floating pointing number of : 0 < number <= 1
Default value: 0.3

#### Customizing the output path

You can change the directory of the generate files by adding `--output-folder $path/to/output`.
For example, `python3 main.py --output-folder ~/generated` will place the generated files under a folder called `generated` under your root directory.
