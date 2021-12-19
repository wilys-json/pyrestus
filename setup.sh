#!/bin/bash

# Install & setup Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> $HOME/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# Install Python3
brew install python3

# Install Dependencies
pip3 install virtualenv virtualenvwrapper &&
python3 -m virtualenv venv &&
source venv/bin/activate &&
pip3 install -r requirements-core.txt &&
python3 -m setup build_ext --inplace -q
