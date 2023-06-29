~/anaconda3/bin/python inference.py --model_name ResNet_object_features --dropout 0.2 --resnet --finetune_only --object_features

~/anaconda3/bin/python inference.py --model_name ResNet_object_scene_features --dropout 0.2 --resnet --object_features --scene_features

~/anaconda3/bin/python inference.py --model_name MobileNet_object_features --dropout 0.2 --mobilenet --finetune_only --object_features

~/anaconda3/bin/python inference.py --model_name MobileNet_object_scene_features --dropout 0.2 --mobilenet --object_features --scene_features

~/anaconda3/bin/python inference.py --model_name ResNet_scene_features --dropout 0.2 --resnet --scene_features

"/media/workstation/0832621B32620DCE/Bala/bala_new_env/bin/python" train_evaluate.py --model_name ResNet_object_features --dropout 0.2 --resnet --finetune_only --object_features
