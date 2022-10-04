#!/bin/bash

# Install & setup Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> $HOME/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# # Install Python3
# brew install python3
#
# # Install Dependencies
# pip3 install virtualenv virtualenvwrapper &&
# python3 -m virtualenv venv &&
# source venv/bin/activate &&
# pip3 install -r requirements-core.txt &&
# pip3 install -r requirements.txt
# python3 -m setup build_ext --inplace -q

# Install Anaconda
brew install --cask anaconda
PREFIX=/usr/local/anaconda3
export PATH="/usr/local/anaconda3/bin:$PATH"

# Install Dependencies
conda env create -f environment.yml

# Activate environment
eval "$(conda shell.bash hook)"
conda activate pyrestus

# Setup
python3 -m setup build_ext --inplace -q
