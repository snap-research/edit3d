ifeq ($(OS),Windows_NT)     # is Windows_NT on XP, 2000, 7, Vista, 10...
	detected_OS := Windows
else
	detected_OS := $(shell uname 2>/dev/null || echo Unknown)
endif

setup_demo:
ifeq ($(detected_OS),Linux)
	conda env list | grep edit3d || conda env create -f environment.yml
	mkdir -p data/models
	curl -L https://www.dropbox.com/s/teez91j76d1pssf/chairs_epoch_2799_iters_280000.pth?dl=0 --output data/models/chairs_epoch_2799_iters_280000.pth
	curl -L https://www.dropbox.com/s/trj8777psawq7dt/airplanes_epoch_2799_iters_156800.pth?dl=0 --output data/models/airplanes_epoch_2799_iters_156800.pth
	curl -L https://www.dropbox.com/s/l2qvhtmma8hr2v9/armchair_10shot_netM_epoch_99.pth?dl=0 --output data/models/armchair_10shot_netM_epoch_99.pth
	curl -L https://www.dropbox.com/s/5rkb6m8ose874wp/sidechair_10shot_netM_epoch_199.pth?dl=0 --output data/models/sidechair_10shot_netM_epoch_199.pth
	curl -L https://www.dropbox.com/s/mrg2lj47n7ilvjc/red_10shot_netM_epoch_99.pth?dl=0 --output data/models/red_10shot_netM_epoch_99.pth
else
	echo "Setup is only configured for linux."
endif

edit_via_scribble:
	python edit3d/edit_via_scribble.py ./config/chair_demo.yaml --imagenum 1 --partid 1

edit_via_sketch:
	python edit_via_sketch.py ./config/airplane_demo.yaml --pretrained=data/models/airplanes_epoch_2799_iters_156800.pth --outdir output/edit_via_sketch --source_dir examples/edit_via_sketch/planes/2e235eafe787ad029a6e43b878d5b335 --epoch 5 --trial 1 --category airplane

reconstruct_sketch:
	python reconstruct_from_sketch.py config/airplane_demo.yaml --pretrained=data/models/airplanes_epoch_2799_iters_156800.pth --outdir output/recon_sketch --impath examples/recon_sketch/airplanes/d54ca25127a15d2b937ae00fead8910d/sketch-T-2.png --mask  --mask-level 0.5

reconstruct_rgb:
	python edit3d/reconstruct_from_rgb.py config/airplane_demo.yaml --pretrained=data/models/airplanes_epoch_2799_iters_156800.pth --outdir output/recon_rgb --impath examples/recon_sketch/airplanes/d54ca25127a15d2b937ae00fead8910d/sketch-T-2.png --mask  --mask-level 0.5

train_plane:
	python train.py config/airplane_train.yaml
