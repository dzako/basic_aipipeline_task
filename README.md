## Sample AI pipeline with object detection designed for CPU

### used libraries

The pipeline code can use any dataset from the Hugging face library.
Also loading local images is possible. I tested on 
the coco dataset https://huggingface.co/datasets/detection-datasets/coco

For object detection the yolov5 network (PyTorch) is applied, 
it has different architectures - I checked two smallest, i.e., yolov5s and yolov5n.
https://github.com/ultralytics/yolov5

### running instructions

1. Install requirements 
`pip install -r requirements.txt`

2. make sure CUDA devices are off
`export CUDA_VISIBLE_DEVICES=""`

3. run pipeline using e.g.
```
python pipeline.py -d detection-datasets/coco -n yolov5n.pt --num_threads 4 --batch 4 --maxim 1000 --imsize 640
```
where `d` is the dataset name (from Hugging face), 
`n` is the network checkpoint name from torch.hub (tested with `yolov5n` and `yolov5s`),
`num_threads` is the number of threads used by PyTorch, `batch` 
is the batch size, `maxim` is the number of images loaded from the dataset,
`imsize` is the size of inferred images.

If you use `detection-datasets/coco` ~20gb dataset will be downloaded from 
Hugging face first, local folder with images can be specified too, in such case 
unzip `coco128.zip` in `local_datasets` and use 
 `-d local_datasets/coco128/images/train2017/`. 

### output

the code will load the specified dataset and infer the object detection using 
the given checkpoint. The images will be outputed to folder named 
`dataset_checkpoint_out` , the profile data will be outputted to 
`dataset_checkpoint_profile`. The output images are just with the bounding box
of localized objects.

Included data from two example experiments (1000 images from coco) using two 
yolo networks: 

```
detection-datasetscoco_yolov5n.pt_profile
```

```
detection-datasetscoco_yolov5s.pt_profile
```

The profile data contains the following text files with profile output:

* `detection_output.txt` the detection output details (identified classes etc..), 
the images contain only the bounding boxes;

* `general_profile.txt` the runtimes of each pipeline step: loading data, 
running the network, dumping labeled images;

* `network_profile.txt` detailed neural network pytorch profiling data, different
operations and layers can be seen there;

* `yolo_profile.txt` breakdown of times spent by yolo on preprocess, 
actual network inference, postprocess;

### optimization considerations

#### Basic optimizations were performed for CPU: 
* a light model was selected (yolov5n weights only 4.1mb);
* model is casted to float `model.to("cpu").float()` 
(further improvement to use ints, but this requires more work);
* images are inferred in batches (batched inference ) - `batch` parameter;

* number of pytorch threads is adjusted `torch.set_num_threads()`;
* The best values I found using preliminary search for 
Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz are `num_threads=4` and `batch=4`;

* other parts of code (data loading / dumping) are timed to make sure there 
are no bottlenecks;

#### Extra steps checked: 
* using onnx runtime checkpoint accelerates further the inference on cpu, 
but is hard to profile.


Wanted to do , but didn't have time at the end:
 * careful tuning of batch/threads,
 * split code into clear modules, 
 * optimize the images dumping part.