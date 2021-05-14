# CS5525_course_project
CS5525 course project: Mushroom classification

## Dataset
Download the dataset from [here](https://drive.google.com/file/d/1sx59m46alf5kZhaRLewZxVq67AGvkxg1/view?usp=sharing) and extract it to the root of this repo

## Python environment
In python3 virtual environment
```sh
pip install requirements.txt
```

## Results
### Run Bag of Visual Words
Construct BoVW features
```
python build_bovw_feature.py
```

Build BovW descriptors
```
python build_bovw_desc.py
```

Test the BoVW classifiers. (You can run this without the first two steps. The results from the first two steps are already saved in ```bovw_result```)
```
python test_bovw.py
```


### Run ResNet
Training
```
python train_resnet.py
```

Testing
```
python test_resnet.py --epoch_checkpoint 70
```



### Run ViT
Training
```
python train_vit.py
```

Testing
```
python test_vit.py --epoch_checkpoint 20
```
