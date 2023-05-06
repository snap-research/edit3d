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
python train.py ./config/airplane_train.yaml --logdir /path/to/save/output
```

### Demo 

#### Setup

Download Pretrained models: [ShapeNet Chairs](https://www.dropbox.com/s/teez91j76d1pssf/chairs_epoch_2799_iters_280000.pth?dl=0), [ShapeNet Airplanes](https://www.dropbox.com/s/trj8777psawq7dt/airplanes_epoch_2799_iters_156800.pth?dl=0)

[./examples](./examples) contains data samples that were used for the following applications of our model. 

#### Shape editing via 2D sketches

```
python edit_via_sketch.py ./config/airplane_demo.yaml --pretrained path/to/pretrained/model --outdir path/to/output --imagelist path/to/target-images --epoch 5 --trial 1 --category airplane 
```

#### Color editing via 2D scribbles 

```
python edit_via_scribble.py ./config/airplane_demo.yaml --pretrained path/to/pretrained/model --outdir path/to/output --imagelist path/to/target-images --epoch 5  --trial 1 --category airplane --partid 3 
```
Note: `--partid` indicates the list of semantic parts where the scribbles are drawn.

#### Shape reconstruction from 2D sketches 

```
python reconstruct_from_sketch.py ./config/airplane_demo.yaml --pretrained path/to/pretrained/model --outdir path/to/output --impath path/to/target-image --epoch 501 --trial 10
```
Note: add `--mask  --mask-level 0.5` to get partial view of the input image

#### Shape reconstruction from RGB images

```
python reconstruct_from_rgb.py ./config/airplane_demo.yaml --pretrained path/to/pretrained/model --outdir path/to/output --impath path/to/target-image --epoch 501 --trial 10
```
Note: add `--mask  --mask-level 0.5` to get partial view of the input image

#### Few-shot shape generation

* Train MineGAN with pretrained MM-VADs
```
python few_shot_adaptation.py ./config/airplane_demo.yaml  --mode train  --pretrained path/to/pretrained-mm-vads --outf path/to/output--niter 200 --nimgs 10 --code shape/or/color --dataset dataset/path
```

* Sample from the adapted MM-VADs
```
python few_shot_adaptation.py ./config/airplane_demo.yaml --mode test --pretrained path/to/pretrained-mineGAN --outf path/to/output    --code shape/or/color 
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
