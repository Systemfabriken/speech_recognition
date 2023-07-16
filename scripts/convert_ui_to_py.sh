#!/bin/bash

# Source the virtual environment
source ../.venv/bin/activate

# Path to the directory containing .ui files
UI_PATH="../src/ui_generated/qt5"
# Path to the directory to place .py files
PY_PATH="../src/ui_generated/pyqt5"

# Ensure output directory exists
mkdir -p $PY_PATH

# For every .ui file in the UI_PATH directory
for ui_file in $UI_PATH/*.ui; do
    # Extract the filename and extension
    filename=$(basename -- "$ui_file")
    extension="${filename##*.}"
    filename="${filename%.*}"
    # Convert the .ui file to a .py file
    py_file="$PY_PATH/$filename.py"
    echo "Converting $ui_file to $py_file"
    pyuic5 -x $ui_file -o $py_file
done 

# Convert resource file
# RESOURCE_FILE="$UI_PATH/assets/resource.qrc"
# RESOURCE_PY_FILE="$PY_PATH/resource_rc.py"
# echo "Converting $RESOURCE_FILE to $RESOURCE_PY_FILE"
# pyrcc5 $RESOURCE_FILE -o $RESOURCE_PY_FILE

