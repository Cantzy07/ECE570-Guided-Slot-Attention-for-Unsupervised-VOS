import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter
from models.feature_aggregation_transformer import FAT
from models.Slot_Attention import GuidedSlotAttention
from transformers import SegformerForImageClassification, SegformerImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self):  
        super(Model, self).__init__()  
        self.rgb_encoder = SegformerForImageClassification.from_pretrained("nvidia/mit-b2").to(device)
        # self.flow_encoder = SegformerForImageClassification.from_pretrained("nvidia/mit-b2")
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b2")
        self.slot_generator = SlotGenerator(in_channels=64, slot_num=2).to(device)
        self.FAT_features = FAT(local_in=128, global_in=16).to(device)
        self.gsa = GuidedSlotAttention().to(device)
        self.encoder_projection = nn.Conv2d(64, 256, kernel_size=1).to(device)

        # Define the refinement CNN block.
        self.cosine_refine = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=1)  # output channels: set to the number of segmentation classes (e.g., 3)
        ).to(device)

    def encode_target(self, image):
        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU

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
    
    def encode_references(self, references):
        # Prepare a list to hold the features from each image
        feature_maps = []

        # Loop through each image
        for image in references:
            # Preprocess the image
            inputs = self.processor(images=image, return_tensors="pt", do_rescale=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU

            # Forward pass through the model
            with torch.no_grad():
                outputs = self.rgb_encoder(**inputs, output_hidden_states=True)

            # Outputs contain multi-scale feature maps
            stage_1 = outputs.hidden_states[0]  # Local (1/4 size)
            stage_2 = outputs.hidden_states[1]  # Local (1/8 size)
            stage_3 = outputs.hidden_states[2]  # Global (1/16 size)
            stage_4 = outputs.hidden_states[3]  # Global (1/32 size)

            # Apply channel-wise softmax: softmax across the channel dimension (dim=0)
            softmax_feature_map = F.softmax(stage_4, dim=1)  # Softmax across channels
            
            # Reduce the channels to a single channel by summing them
            mj_gt = softmax_feature_map.sum(dim=1, keepdim=True)  # Sum over channels, keeping the spatial dimensions

            # You can choose which stage to use, here I am using stage_1 (local, 1/4 size)
            feature_maps.append(mj_gt)  # Add the feature map to the list

        # Stack the feature maps along the channel dimension
        # Assuming feature_maps is a list of tensors with shape [B, C, H, W]
        combined_features = torch.cat(feature_maps, dim=1)  # Concatenate along the channel dimension

        return combined_features
    
    def cosine_similarity_decoding(self, X_L, P_Sr):
        """
        X_L: Original Image [B, C, H, W] torch.Size([64, 128, 128])
        P_Sr: Refined slots [B, num_slots, slot_dim] torch.Size([1, 2, 256])
        P_A: Aggregated features [B, C', H', W'] torch.size([1, 1, 16, 16])
        """
        # Flatten spatial dimensions
        # to_tensor = transforms.ToTensor()
        X_L = self.encoder_projection(X_L)
        X_flat = X_L.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        P_Sr = P_Sr.transpose(1, 2)  # [B, slot_dim, num_slots]
        P_Srf = P_Sr[:, :, 0]
        P_Srb = P_Sr[:, :, 1]

        # print("x ps pa", X_flat.shape, P_Sr.shape, P_A.shape

        CM_sf = F.cosine_similarity(
            X_flat.unsqueeze(2),  # [B, H*W, 1, C]
            P_Srf.unsqueeze(1),    # [B, 1, slot_dim, num_slots]
            dim=-1
        )  # [B, H*W, num_slots]

        CM_sb = F.cosine_similarity(
            X_flat.unsqueeze(2),  # [B, H*W, 1, C]
            P_Srb.unsqueeze(1),    # [B, 1, slot_dim, num_slots]
            dim=-1
        )  # [B, H*W, num_slots]]

        similarity_map_sf = CM_sf.view(X_L.shape[0], X_L.shape[2], X_L.shape[3], -1)
        similarity_map_sf = similarity_map_sf.permute(0, 3, 1, 2)  # [B, num_slots, H, W]

        similarity_map_sb = CM_sb.view(X_L.shape[0], X_L.shape[2], X_L.shape[3], -1)
        similarity_map_sb = similarity_map_sb.permute(0, 3, 1, 2)  # [B, num_slots, H, W]

        # Combine aggregator, foreground, background
        combined_sim_map = torch.cat(
            [similarity_map_sf, similarity_map_sb],
            dim=1
        )  # [B, 3, H, W]
        
        # Turn them into a probability distribution across the 3 “slots” for each pixel
        masks = F.softmax(combined_sim_map, dim=1)  # [B, 3, H, W]
        masks = self.cosine_refine(masks)

        # print("mask", masks.shape)
        return masks
    
    def forward(self, x_target, x_references):
        # using stage_1 as local features, stage_3 as X_L encoder features, stage_4 as global features
        # Stage 1 (Local) Shape: torch.Size([1, 64, 128, 128])
        # Stage 2 (Local) Shape: torch.Size([1, 128, 64, 64])
        # Stage 3 (Global) Shape: torch.Size([1, 320, 32, 32])
        # Stage 4 (Global) Shape: torch.Size([1, 512, 16, 16])
        stage_1, stage_2, stage_3, stage_4 = self.encode_target(x_target)
        references_features = self.encode_references(x_references)

        # Slot Output Shape: torch.Size([1, 2, 128, 128])
        slots = self.slot_generator(stage_1)

        # Output shape: torch.Size([1, 1, 16, 16])
        aggregated_features = self.FAT_features(stage_1, references_features)

        # gsa shape torch.Size([1, 2, 256])
        refined_slots = self.gsa(aggregated_features, slots)

        # mask shape torch.Size([1, 3, 64, 64])
        x = self.cosine_similarity_decoding(stage_1, refined_slots)
        return x

class SlotGenerator(nn.Module):
    def __init__(self, in_channels, slot_num):
        super(SlotGenerator, self).__init__()  
        self.conv = nn.Conv2d(in_channels, slot_num, kernel_size=1).to(device)  # [B, slot_num, H, W]
        self.softmaxh = nn.Softmax(dim=-1)
        self.softmaxw = nn.Softmax(dim=-2)

    def forward(self, x):
        x = self.conv(x)
        x = self.softmaxh(x)
        x = self.softmaxw(x)
        return x