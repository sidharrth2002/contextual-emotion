# ~/anaconda3/bin/python train_all.py --model_name ViT_object_features --dropout 0.4 --vit --object_features

~/anaconda3/bin/python train_all.py --model_name ViT_scene_features --dropout 0.4 --vit --scene_features --finetune_only

~/anaconda3/bin/python train_all.py --model_name ViT_object_scene_features --dropout 0.4 --vit --object_features --scene_features

~/anaconda3/bin/python train_all.py --model_name ResNet_object_features --dropout 0.2 --resnet --object_features

~/anaconda3/bin/python train_all.py --model_name ResNet_scene_features --dropout 0.2 --resnet --scene_features

~/anaconda3/bin/python train_all.py --model_name ResNet_object_scene_features --dropout 0.2 --resnet --object_features --scene_features

~/anaconda3/bin/python train_all.py --model_name MobileNet_object_features --dropout 0.2 --mobilenet --object_features

~/anaconda3/bin/python train_all.py --model_name MobileNet_scene_features --dropout 0.2 --mobilenet --scene_features

~/anaconda3/bin/python train_all.py --model_name MobileNet_object_scene_features --dropout 0.2 --mobilenet --object_features --scene_features

~/anaconda3/bin/python train_all.py --model_name EffNet_object_features --dropout 0.4 --efficientnet --object_features

~/anaconda3/bin/python train_all.py --model_name EffNet_scene_features --dropout 0.4 --efficientnet --scene_features

~/anaconda3/bin/python train_all.py --model_name EffNet_object_scene_features --dropout 0.4 --efficientnet --object_features --scene_features


