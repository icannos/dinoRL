

# Dino Deep Q Learning (DDQL)

## Requirements

```
Opencv cv2
Keras
Tensorflow
Numpy
```

## Usage

### Training

For training one can use `nsteps_dino.py` or `train_dino.py`. The first one uses the k-step DQL algorithm and the latter
uses the single step DQL algorithm.

### Demo

Any trained model can be executed by the `demo.py` script. It only takes as argument the path of the model file.