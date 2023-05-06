ifeq ($(OS),Windows_NT)     # is Windows_NT on XP, 2000, 7, Vista, 10...
	detected_OS := Windows
else
	detected_OS := $(shell uname 2>/dev/null || echo Unknown)
endif

setup_demo:
ifeq ($(detected_OS),Linux)
    conda activate edit3d || conda env create -f environment.yml
	mkdir -p data/models
	curl -L https://www.dropbox.com/s/teez91j76d1pssf/chairs_epoch_2799_iters_280000.pth?dl=0 --output data/models/chairs_epoch_2799_iters_280000.pth
	curl -L https://www.dropbox.com/s/trj8777psawq7dt/airplanes_epoch_2799_iters_156800.pth?dl=0 --output data/models/airplanes_epoch_2799_iters_156800.pth
else
	echo "Setup is only configured for linux."
endif

reconstruct_sketch_demo:
	python reconstruct_from_sketch.py config/airplane_demo.yaml --pretrained=data/models/airplanes_epoch_2799_iters_156800.pth --outdir output --impath examples/recon_sketch/airplanes/d54ca25127a15d2b937ae00fead8910d/sketch-T-2.png --mask  --mask-level 0.5
