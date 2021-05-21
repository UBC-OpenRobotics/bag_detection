# Bag Detection

This repository implements a script that executes bag detection based on a trained tiny YOLOv4 model.

![alt text](https://github.com/UBC-OpenRobotics/bag_detection/blob/main/output/bag_detection_sample.gif "Example of Bag Detection")


## Usage
To run inference,
```
python3 infer_yolo.py -i path/to/image/or/dir -s
```
The `-s` flag will enable visualization.

The default output is the bounding box and label for each input image.

```
../bag_detection_bak/test/bags_1.jpg: Label: bag	Bbox: (23,-40,392,563)
```
