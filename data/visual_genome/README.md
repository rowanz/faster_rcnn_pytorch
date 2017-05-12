# Dataset Preparation
Adapted from [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md).

This instruction contains details about how to convert the VisualGenome dataset into a format that can be read by the framework. Alternatively,
you may follow the instruction to download a pre-processed dataset.

## Overview

A dataset for the framework consists of five files:
1. An image database in hdf5 format.
2. An scene graph database in hdf5 format.
3. An scene graph database metadata file in json format.
4. An RoI proposal database in hdf5 format.
5. An RoI distribution for normalizing the bounding boxes.

**Important:** Note that (1) takes ~320GB of space. Hence we recommend creating (1) by yourself and download (2-5). If you just want to test/visualize predictions,
you may download a subset of the processed dataset following the instructions in the next section.
Also note that the framework does not include a regional proposal network implementation. Hence (4) is needed to run the framework.

## Download pre-processed dataset.
You may download the pre-processd full VG dataset using the following links
1. Image database (currently unavailable, please refer to the next section on how to create an IMDB by yourself)
2. Scene graph database: [VG-SGG.h5](http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG.h5)
3. Scene graph database metadata: [VG-SGG-dicts.json](http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG-dicts.json)
4. RoI proposals: [proposals.h5](http://cvgl.stanford.edu/scene-graph/dataset/proposals.h5)
5. RoI distribution: [bbox_distribution.npy](http://cvgl.stanford.edu/scene-graph/dataset/bbox_distribution.npy)


## Convert VisualGenome to desired format
(i). To start with, download VisualGenome dataset using the following links:
- Images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Put the images in the folder `data/visual_genome/VG_100K`
- [Image metadata](http://cvgl.stanford.edu/scene-graph/VG/image_data.json) Put the metadata under `data/visual_genome/datasetv1_2`
- [VG scene graph](http://cvgl.stanford.edu/scene-graph/VG/VG-scene-graph.zip) Extract this to `data/visual_genome/scene_graphs`

(iii). Create image database + ROI database by going to the directory `data/visual_genome/preprocess_code` and running
```
python vg_to_imdb.py
./create_roidb.sh
```
Then, move `imdb_1024.h5`, 'VG-SGG.h5`, and `VG-SGG-dicts.json` up one level, into `visual_genome/`

(v). Use the script provided by py-faster-rcnn to generate (4).

(vi). Change line 93 of `tools/train_net.py` to `True` to generate (5).

(vii). Finally, place (1-5) in `data/visual_genome`.

```
data/visual_genome/imdb_1024.h5
data/visual_genome/bbox_distribution.npy
data/visual_genome/proposals.h5
data/visual_genome/VG-SGG-dicts.json
data/visual_genome/VG-SGG.h5
```
