import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter
import numpy as np
from transformers import SegformerForImageClassification, SegformerImageProcessor
from sklearn.cluster import KMeans
import torch
from FAT import FAT

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is None:
                pass
            elif m.bias is not None:
                nn.init.zeros_(m.bias)
            else:
                nn.init.ones_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.Upsample, Parameter, nn.AdaptiveAvgPool2d, nn.Sigmoid)):
            pass
        else:
            try:
                m.initialize()
            except:
                pass

class Model(nn.Module):
    def __init__(self):  
        super(Model, self).__init__()  
        self.rgb_encoder = SegformerForImageClassification.from_pretrained("nvidia/mit-b2")
        self.flow_encoder = SegformerForImageClassification.from_pretrained("nvidia/mit-b2")
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b2")
        self.FAT_features = FAT(local_in=128, global_in=16)
        self.FAT_slots = FAT(local_in=128, global_in=16)

    def encode_image(self, image):
        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt")

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.rgb_encoder(**inputs, output_hidden_states=True)

        # Outputs contain multi-scale feature maps
        # B, C, H, W
        stage_1 = outputs.hidden_states[0]  # Local (1/4 size)
        stage_2 = outputs.hidden_states[1]  # Local (1/8 size)
        stage_3 = outputs.hidden_states[2]  # Global (1/16 size)
        stage_4 = outputs.hidden_states[3]  # Global (1/32 size)

        return stage_1, stage_2, stage_3, stage_4
    
    def generate_slots(self, image):
        stage_1, stage_2, stage_3, stage_4 = self.encode_image(image)

        slot_generator = SlotGenerator(in_channels=64, slot_num=2)

        slots = slot_generator(stage_1)

        return slots
    
    # def forward(self, x):


class SlotGenerator(nn.Module):
    def __init__(self, in_channels, slot_num):
        super(SlotGenerator, self).__init__()  
        self.input_1x1conv = nn.Conv2d(in_channels=in_channels, out_channels=slot_num, kernel_size=1)
        # Pixel-wise softmax
        self.softmax_w = nn.Softmax(dim=-1)  # Apply softmax along the last dimension (width)
        self.softmax_h = nn.Softmax(dim=-2)  # Apply softmax along the second last dimension (height)
        self.globalavgpool = nn.AvgPool2d(kernel_size=(128, 128))  # Pool over the entire spatial dimension

    def forward(self, x):
        x = self.input_1x1conv(x)
        x = self.softmax_w(x)
        x = self.softmax_h(x)
        x = self.globalavgpool(x)
        return x