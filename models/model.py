import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter
import numpy as np
import torchvision
from transformers import SegformerForImageClassification, SegformerImageProcessor
from sklearn.cluster import KMeans
import torch

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

def get_cluster_mask(self):
        # Reshape to (num_pixels, num_features)
        C, H, W = self.embedding.shape[1:]  
        features = self.embedding.view(C, H * W).T.cpu().numpy()  # Shape: (H*W, C)
        self.kmeans.fit(self.embedding)
        cluster_labels = self.kmeans.labels_.reshape(H, W)  # Reshape back to spatial dimensions
        # Generate D binary masks
        cluster_masks = np.zeros((len(cluster_labels), H, W))  # Shape: (D, H, W)

        for d in range(len(cluster_labels)):
            cluster_masks[d] = (cluster_labels == d).astype(np.uint8)  # 1 for pixels in cluster d, 0 otherwise

        return cluster_masks

def gwap(masks, embedding):
    features = []
    for mask in masks:
        feature_vector = []
        for channel in embedding:
            weighted_feature_map = np.multiply(channel, mask)
            weighted_sum = sum(weighted_feature_map)
            mask_weight_sum = sum(mask)
            weighted_avg = weighted_sum / mask_weight_sum
            feature_vector.append(weighted_avg)
        
        features.append(feature_vector)

    return features


class Model(nn.Module):
    def __init__(self):    
        self.rgb_encoder = SegformerForImageClassification.from_pretrained("nvidia/mit-b2")
        self.flow_encoder = SegformerForImageClassification.from_pretrained("nvidia/mit-b2")
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b2")

class LocalExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.num_clusters = 64
        self.input_1x1conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.pooling = nn.AvgPool2d()

    def get_cluster_maps(self, num_clusters, encoded_features):
        kmeans = KMeans(n_clusters=num_clusters, mode='Euclidean', max_iter=10, verbose=0)
        cluster_labels = kmeans.fit_predict(encoded_features)

        # Reshape cluster labels back to (H, W)
        cluster_map = cluster_labels.reshape(encoded_features.size(1), encoded_features.size(2))

        # Initialize a list to store binary cluster maps
        binary_cluster_maps = []

        # Create binary maps for each cluster
        for cluster_id in range(num_clusters):
            binary_map = (cluster_map == cluster_id).astype(np.float32)
            binary_cluster_maps.append(binary_map)

        # Convert list of binary maps to a tensor (optional)
        binary_cluster_maps = torch.tensor(np.stack(binary_cluster_maps, axis=0))

        return binary_cluster_maps
    
    def forward(self, x):
        x = self.input_1x1conv(x)
        x = get_cluster_mask(self.num_clusters, x)
        x = self.pooling(x)
        return x

class GlobalExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.fpn = torchvision.ops.FeaturePyramidNetwork(in_channels_list=[64, 128, 320, 512], out_channels=256)
        self.ref_1x1conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.softmax_global = nn.Softmax2d()
        self.pooling = nn.AvgPool2d()

    def forward(self, x):
        x = self.fpn(x)
        x = self.ref_1x1conv
        x = self.softmax_global(x)
        x = self.pooling(x)
        return x

class SlotGenerator(nn.Module):
    def __init__(self, in_channels, slot_num):
        self.input_1x1conv = nn.Conv2d(in_channels=in_channels, out_channels=slot_num, kernel_size=1)
        # Pixel-wise softmax
        self.softmax_w = nn.Softmax(dim=-1)  # Apply softmax along the last dimension (width)
        self.softmax_h = nn.Softmax(dim=-2)  # Apply softmax along the second last dimension (height)
        self.pooling = nn.AvgPool2d()

    def forward(self, x):
        x = self.input_1x1conv(x)
        x = self.softmax_w(x)
        x = self.softmax_h(x)
        x = self.pooling(x)
        return x