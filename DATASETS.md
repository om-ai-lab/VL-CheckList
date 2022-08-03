# How to install datasets

The datasets we use are listed below:

- Visual-genome [Link](https://visualgenome.org/api/v0/api_home.html)
- HAKE [Link](http://hake-mvig.cn/home/)
- Visual Attributes in the Wild (VAW) [Link](https://vawdataset.com/)
- SWiG [Link](https://github.com/allenai/swig)

As the images in the VAW dataset come from the Visual Genome dataset, we only need three picture folders. The file structure looks like below:
```
$DATASET/
|–– hake/
|–– swig/
|–– vg/
```

## HAKE
- Create a folder named `hake/` under `$DATASET/`.
- Create `hake_images_20190730/`,`hake_images_20200614/`,`hcvrd/`,`hico_20160224_det/`,`openimages/`,`pic/` and `vcoco/` under `hake/`.
- Download the images from the [dataset link](https://github.com/DirtyHarryLYL/HAKE/tree/master/Images#download-images-for-hake).
- Put the images in the corresponding folder.

The folder tree looks like:
```
hake/
├── hake_images_20190730
│   └── xxx.jpg
├── hake_images_20200614
│   └── xxx.jpg
├── hcvrd
│   └── xxx.jpg
├── hico_20160224_det
│   └── images
│       ├── test2015
│       │   └── xxx.jpg
│       └── train2015
│           └── xxx.jpg
├── openimages
│   └── xxx.jpg
├── pic
│   └── xxx.jpg
└── vcoco
    ├── train2014
    |    └── xxx.jpg
    └── val2014
        └── xxx.jpg
```
## SWiG
- Create a folder named `swig/` under `$DATASET/`.
- Download the images from the [dataset link](https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip).

The folder tree looks like:
```
swig/
└── xxx.jpg
```
## VG & VAW
- Create a folder named `vg/` under `$DATASET/`.
- Create `VG_100K/` and `VG_100K_2/` under `vg/`.
- Download the images from the [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip).
- Put the images in the corresponding folder.

The folder tree looks like:
```
vg/
|–– VG_100K/
    └── xxx.jpg
|–– VG_100K_2/
    └── xxx.jpg
```

