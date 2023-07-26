# Define variables
SHELL := /bin/bash
SCRIPT_PATH := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
VENV_PATH := $(SCRIPT_PATH)/.venv/bin/activate
PROMOTIONS_PATH := $(SCRIPT_PATH)/src/qt_promotions
PYQT5_RESOURCE_PATH := $(SCRIPT_PATH)/src/ui_generated/pyqt5
SITE_PACKAGES_PATH := $(shell source $(VENV_PATH) && python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
PTH_FILE_PATH := $(SITE_PACKAGES_PATH)/qt_promotions.pth
PYQT5_RESOURCE_PTH := $(SITE_PACKAGES_PATH)/pyqt5-resource-file.pth

# Default target executed when no arguments given to make.
default_target: build
.PHONY: default_target

# Target for creating environment
environment: download_models
	# Install Python3 and pip3
	sudo apt update
	sudo apt install -y python3 python3-pip python3-venv

	# Install Qt5 and Qt Designer
	sudo apt install -y qttools5-dev-tools

	# Install other dependencies
	sudo apt-get install -y libxcb-xinerama0 portaudio19-dev ffmpeg python3-av libavdevice-dev libavdevice59

	unset GTK_PATH

	# Create a Python virtual environment
	python3 -m venv .venv
	source $(VENV_PATH)

	# Install the necessary Python libraries inside the virtual environment
	source $(VENV_PATH) && pip3 install pyqt5 pyqt5-tools opencv-python-headless dlib face_recognition scikit-learn sounddevice openai-whisper pyaudio torch torchaudio IPython matplotlib transformers

.PHONY: environment

# Target for downloading the latest tflite model and scorer
download_models:
	mkdir -p $(SCRIPT_PATH)/models

	# Download the latest tflite model
	wget "https://github.com/coqui-ai/STT-models/releases/download/english%2Fcoqui%2Fv1.0.0-huge-vocab/model.tflite" -O $(SCRIPT_PATH)/models/model.tflite

	# Download the latest scorer
	wget "https://github.com/coqui-ai/STT-models/releases/download/english%2Fcoqui%2Fv1.0.0-huge-vocab/huge-vocabulary.scorer" -O $(SCRIPT_PATH)/models/huge-vocabulary.scorer

.PHONY: download_models

# Target for building
build:
	# cd $(SCRIPT_PATH)/scripts && ./convert_ui_to_py.sh
	# cd $(SCRIPT_PATH)/rope-robot-drm-protocol && make python
	# echo $(PROMOTIONS_PATH) > $(PTH_FILE_PATH)
	# echo $(PYQT5_RESOURCE_PATH) > $(PYQT5_RESOURCE_PTH)
.PHONY: build

# Target for running the main script
run: build
	source $(VENV_PATH) && python3 src/main.py
.PHONY: run_face_detection

# Target for running tests
run_tests: build
.PHONY: run_tests

run_hw_tests: build
.PHONY: run_hw_tests
