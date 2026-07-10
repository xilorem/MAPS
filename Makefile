PYTHON ?= ./.venv/bin/python
MAPS_IR_DIR ?= maps-ir

GENERATED_DIR := generated
PIPELINE_JSON := $(GENERATED_DIR)/magia_example.pipeline.json

.PHONY: all magia-example pipeline-json maps-translate maps-mlir magia-header magia-data clean-generated

all: magia-example

magia-example: pipeline-json
	$(MAKE) -C $(MAPS_IR_DIR) magia-example PIPELINE_JSON=../$(PIPELINE_JSON) GENERATED_DIR=../$(GENERATED_DIR)

pipeline-json:
	$(PYTHON) examples/magia_example.py

maps-translate:
	$(MAKE) -C $(MAPS_IR_DIR) maps-translate

maps-mlir magia-header magia-data: pipeline-json
	$(MAKE) -C $(MAPS_IR_DIR) $@ PIPELINE_JSON=../$(PIPELINE_JSON) GENERATED_DIR=../$(GENERATED_DIR)

clean-generated:
	$(MAKE) -C $(MAPS_IR_DIR) clean-generated GENERATED_DIR=../$(GENERATED_DIR)
