## Cross-Modal 3D Shape Generation and Manipulation (ECCV 2022)

This repository contains the source code for the ECCV 2022 paper <u>Cross-Modal 3D Shape Generation and Manipulation</u>. Our implementation is based on [DualSDF](https://www.cs.cornell.edu/~hadarelor/dualsdf/). 

[[Project page]](https://people.cs.umass.edu/~zezhoucheng/edit3d)  

Note: (07/18/2022) This codebase has not yet been systematically tested. We're working in progress. Stay tuned!

### Installation

```
conda env create -f environment.yml
source activate edit3d
```

### Training Multi-Modal Variational Auto-Decoders (MM-VADs)
```
python edit3d/train.py ./config/airplane_train.yaml --logdir /path/to/save/output
```
or 
```
make train
```

### Demo 

#### Setup

Download Pretrained models: [ShapeNet Chairs](https://www.dropbox.com/s/teez91j76d1pssf/chairs_epoch_2799_iters_280000.pth?dl=0), [ShapeNet Airplanes](https://www.dropbox.com/s/trj8777psawq7dt/airplanes_epoch_2799_iters_156800.pth?dl=0)

[./examples](./examples) contains data samples that were used for the following applications of our model. 

#### Shape editing via 2D sketches

```
python edit3d/edit_via_sketch.py ./config/airplane_demo.yaml --pretrained path/to/pretrained/model --outdir path/to/output --source_dir path/to/target-images --epoch 5 --trial 1 --category airplane 
```
or 
```
make edit_via_sketch
```
#### Color editing via 2D scribbles 

For chair: 
```
python edit3d/edit_via_scribble.py ./config/chair_demo.yaml --imagenum 1 --partid 1

```

For airplane: 
```
python edit3d/edit_via_scribble.py ./config/airplane_demo.yaml --imagenum 1 --partid 1

```
```
partid
    for chairs:
        1: seat
        2: seat+arm
        3: seat+back

    for airplane:
        1: body only
        2: body+wings

```
`--save_mesh`: sets whether to save 3d mesh

`--imagenum`: 1: chair, 2: couch chair, 3,4: airplanes

`--colors`: 0: random, 1:blue+lime, 2: red+blue, 3: magenta+lightblue 

#### Shape reconstruction from 2D sketches 

```
python edit3d/reconstruct_from_sketch.py ./config/airplane_demo.yaml --pretrained path/to/pretrained/model --outdir path/to/output --impath path/to/target-image --epoch 501 --trial 10
```
Note: add `--mask  --mask-level 0.5` to get partial view of the input image
or 
```
make reconstruct_sketch
```
#### Shape reconstruction from RGB images

```
python edit3d/reconstruct_from_rgb.py ./config/airplane_demo.yaml --pretrained path/to/pretrained/model --outdir path/to/output --impath path/to/target-image --epoch 501 --trial 10
```
or
```
make reconstruct_rgb
```
Note: add `--mask  --mask-level 0.5` to get partial view of the input image

#### Few-shot shape generation

* Train MineGAN with pretrained MM-VADs
```
python edit3d/few_shot_adaptation.py ./config/airplane_demo.yaml  --mode train  --pretrained path/to/pretrained-mm-vads --outf path/to/output--niter 200 --nimgs 10 --code shape/or/color --dataset dataset/path
```

* Sample from the adapted MM-VADs
```
python edit3d/few_shot_adaptation.py ./config/airplane_demo.yaml --mode test --pretrained path/to/pretrained-mineGAN --outf path/to/output    --code shape/or/color 
```

* Download pretrained MineGAN: 

These models are trained with 10 RGB examples per category: 
[Armchairs](https://www.dropbox.com/s/l2qvhtmma8hr2v9/armchair_10shot_netM_epoch_99.pth?dl=0), [Side chairs](https://www.dropbox.com/s/5rkb6m8ose874wp/sidechair_10shot_netM_epoch_199.pth?dl=0), [Pink chairs](https://www.dropbox.com/s/mrg2lj47n7ilvjc/red_10shot_netM_epoch_99.pth?dl=0)


### Citation

```
@inproceedings{Cheng2022-edit3d,
author = {Zezhou Cheng and Menglei Chai and Jian Ren and Hsin-Ying Lee and Kyle Olszewski and Zeng Huang and Subhransu Maji and Sergey Tulyakov},
title = {Cross-Modal 3D Shape Generation and Manipulation},
booktitle = {European Conference on Computer Vision (ECCV) },
year = {2022}
}
```
