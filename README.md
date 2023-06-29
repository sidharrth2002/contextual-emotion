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