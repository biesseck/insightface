# Face Detection with RetinaFace

### Main requirements
- Python 3.9
- CUDA 11.2

### Config environment
```
git clone https://github.com/biesseck/insightface.git
cd insightface/detection

ENV=retinaface_insightface_py39
conda env create -y --name $ENV --file environment.yml
conda activate $ENV
pip3 install -r requirements.txt
```

### Compile RCNN
```
cd retinaface
make
```

### Download pre-trained model RetinaFace-R50

- Save file [retinaface-R50.zip](https://drive.google.com/file/d/1_DKgGxQWqlTqe78pw0KavId9BIMNUWfu/view?usp=sharing) to folder `retinaface/model`
```
cd retinaface/model
unzip retinaface/model/retinaface-R50.zip -d retinaface/model/retinaface-R50
```

### Run face detection script
```
python detect_crop_faces_retinaface.py --input_path /path/to/dataset/root/folder --align_face --save_crops --process_only_biggest_face
```
- The following directory will be created:
  ```
  ├─ /path/to/dataset/root/folder
      ├─ subdir0
          ├─ validation_images_DETECTED_FACES_RETINAFACE_scales=[1.0]
              ├─ imgs
              ├─ txt
              ├─ files_no_face_detected_thresh=0.01_starttime=2024-03-26_22-21-07.txt
  
  ```




</br></br></br></br></br>


## Face Detection

<div align="left">
  <img src="https://insightface.ai/assets/img/custom/logo3.jpg" width="240"/>
</div>


## Introduction

These are the face detection methods of [InsightFace](https://insightface.ai)


<div align="left">
  <img src="https://insightface.ai/assets/img/github/11513D05.jpg" width="800"/>
</div>


### Datasets

  Please refer to [datasets](_datasets_) page for the details of face detection datasets used for training and evaluation.

### Evaluation

  Please refer to [evaluation](_evaluation_) page for the details of face recognition evaluation.


## Methods


Supported methods:

- [x] [RetinaFace (CVPR'2020)](retinaface)
- [x] [SCRFD (Arxiv'2021)](scrfd)
- [x] [blazeface_paddle](blazeface_paddle)


## Contributing

We appreciate all contributions to improve the face detection model zoo of InsightFace. 


