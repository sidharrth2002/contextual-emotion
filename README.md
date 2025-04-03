# Context-Aware Multi-Stream Networks for Dimensional Emotion Prediction in Images

Teaching machines to comprehend the nuances of emotion from photographs is a particularly challenging task. Emotion perception— naturally a subjective problem, is often simplified for computational purposes into categorical states or valence-arousal dimensional space, the latter being a lesser-explored problem in the literature. This paper proposes a multi-stream context-aware neural network model for dimensional emotion prediction in images. Models were trained using a set of object and scene data along with deep features for valence, arousal, and dominance estimation. Experimental evaluation on a large-scale image emotion dataset demonstrates the viability of our proposed approach. Our analysis postulates that the understanding of the depicted object in an image is vital for successful predictions whilst relying on scene information can lead to somewhat confounding effects.

### GradCam Heatmaps

<p align="center">

<img src="./gradcam.jpg" width="400" />

</p>

### Folder Structure

```
|-- benchmarking
|-- decisions
    |-- image_model
    |-- mlp_partial_dataset
    |-- object_data_only
    |-- remove-0-rows
    |-- without_rounding
|-- gradcam_heatmaps
|-- generate_graphs.py
|-- infer.sh
|-- train.sh
|-- train_all.py
|-- train_evaluate.py
```

For enquiries related to the code, please contact [Sidharrth](mailto:sidharrth2002@gmail.com).

### Citation

If you find our code or algorithm useful for your research, please cite:

```
@inproceedings{10221960,
  title        = {Context-Aware Multi-Stream Networks for Dimensional Emotion Prediction in Images},
  author       = {Nagappan, Sidharrth and Tan, Jia Qi and Wong, Lai-Kuan and See, John},
  year         = 2023,
  booktitle    = {2023 IEEE International Conference on Image Processing (ICIP)},
  pages        = {2480--2484},
  doi          = {10.1109/ICIP49359.2023.10221960},
}
```
