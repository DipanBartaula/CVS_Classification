"""
VideoMAE Model for CVS (Critical View of Safety) Classification.

Wraps the HuggingFace VideoMAEForVideoClassification with a custom
multi-label classification head for 3 binary CVS criteria.
"""
import logging
import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEConfig
from typing import Dict, Optional, Tuple

import config

logger = logging.getLogger("CVS_Classification")


class VideoMAECVSClassifier(nn.Module):
    """
    VideoMAE-based multi-label classifier for CVS assessment.
    
    Architecture:
        VideoMAE Encoder (frozen/fine-tuned) -> [CLS] token
        -> LayerNorm -> Dropout -> FC (768 -> 256) -> GELU -> Dropout
        -> FC (256 -> 3) -> Sigmoid
    
    Each output neuron corresponds to one CVS criterion:
        - C1: Two Structures (cystic duct & artery)
        - C2: Hepatocystic Triangle Dissection
        - C3: Cystic Plate Separation
    
    Args:
        model_name: HuggingFace model identifier for pretrained VideoMAE.
        num_classes: Number of CVS criteria (default: 3).
        dropout_rate: Dropout probability in the classification head.
        freeze_backbone: If True, freeze all encoder parameters.
        freeze_layers: Number of encoder layers to freeze (from bottom).
                       Only used if freeze_backbone is False.
    """
    
    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
        freeze_layers: int = 0,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained VideoMAE
        logger.info(f"Loading pretrained VideoMAE: {model_name}")
        
        # Load the base model config and modify for our use
        mae_config = VideoMAEConfig.from_pretrained(model_name)
        mae_config.num_labels = num_classes
        
        # Load pretrained model (ignore classification head mismatch)
        self.videomae = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            config=mae_config,
            ignore_mismatched_sizes=True,
        )
        
        # Get the hidden size from the encoder
        hidden_size = mae_config.hidden_size  # 768 for base, 1024 for large
        
        # Replace the default classification head with our custom multi-label head
        self.videomae.classifier = nn.Identity()  # Remove default head
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes),
        )
        
        # Initialize classifier weights
        self._init_classifier()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        elif freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        # Log model info
        total, trainable = self._count_params()
        logger.info(
            f"VideoMAECVSClassifier initialized:"
            f"\n  Hidden size: {hidden_size}"
            f"\n  Num classes: {num_classes}"
            f"\n  Total params: {total:,}"
            f"\n  Trainable params: {trainable:,}"
            f"\n  Frozen params: {total - trainable:,}"
        )
    
    def _init_classifier(self):
        """Initialize classification head with Xavier uniform."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _freeze_backbone(self):
        """Freeze all VideoMAE encoder parameters."""
        for param in self.videomae.videomae.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen: only classifier head is trainable")
    
    def _freeze_layers(self, num_layers: int):
        """Freeze the first N encoder layers."""
        layers = self.videomae.videomae.encoder.layer
        for i, layer in enumerate(layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        logger.info(f"Frozen first {num_layers}/{len(layers)} encoder layers")
    
    def _count_params(self) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            pixel_values: Video frames tensor of shape (B, T, C, H, W)
            labels: Optional ground truth labels of shape (B, num_classes)
        
        Returns:
            Dict with:
                - logits: Raw predictions of shape (B, num_classes)
                - probabilities: Sigmoid probabilities of shape (B, num_classes)
                - loss: (optional) BCE loss if labels provided
                - features: CLS token features of shape (B, hidden_size)
        """
        # Get VideoMAE encoder output
        outputs = self.videomae(pixel_values=pixel_values, output_hidden_states=True)
        
        # The model returns logits from its (now Identity) classifier
        # We need the hidden states to get the CLS features
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # Use the last hidden state's CLS token (first token)
            last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_size)
            # Use mean pooling over all tokens (more robust than CLS for VideoMAE)
            features = last_hidden.mean(dim=1)  # (B, hidden_size)
        else:
            # Fallback: run through encoder directly
            encoder_outputs = self.videomae.videomae(pixel_values=pixel_values)
            sequence_output = encoder_outputs.last_hidden_state  # (B, seq_len, hidden_size)
            features = sequence_output.mean(dim=1)  # (B, hidden_size)
        
        # Classification head
        logits = self.classifier(features)  # (B, num_classes)
        probabilities = torch.sigmoid(logits)  # (B, num_classes)
        
        result = {
            "logits": logits,
            "probabilities": probabilities,
            "features": features,
        }
        
        # Compute loss if labels are provided
        if labels is not None:
            result["loss"] = self.compute_loss(logits, labels)
        
        return result
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Binary Cross-Entropy loss for multi-label classification.
        
        Args:
            logits: Raw predictions (B, num_classes)
            labels: Ground truth (B, num_classes)
            pos_weight: Optional positive class weights (num_classes,)
        
        Returns:
            Scalar loss value.
        """
        if pos_weight is not None:
            pos_weight = pos_weight.to(logits.device)
        
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return loss_fn(logits, labels)
    
    def predict(
        self,
        pixel_values: torch.Tensor,
        threshold: float = config.CLASSIFICATION_THRESHOLD,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference and return binary predictions.
        
        Args:
            pixel_values: Video frames (B, T, C, H, W)
            threshold: Classification threshold for binary decision
        
        Returns:
            Dict with probabilities and binary predictions
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(pixel_values)
        
        predictions = (result["probabilities"] >= threshold).int()
        result["predictions"] = predictions
        
        # Determine if CVS is achieved (all 3 criteria met)
        result["cvs_achieved"] = (predictions.sum(dim=1) == self.num_classes).int()
        
        return result


def build_model(
    model_name: str = config.MODEL_NAME,
    num_classes: int = config.NUM_CLASSES,
    freeze_backbone: bool = False,
    freeze_layers: int = 0,
    dropout_rate: float = 0.3,
) -> VideoMAECVSClassifier:
    """
    Factory function to build the VideoMAE CVS classifier.
    
    Args:
        model_name: HuggingFace pretrained model name.
        num_classes: Number of output classes (CVS criteria).
        freeze_backbone: Whether to freeze the entire backbone.
        freeze_layers: Number of layers to freeze from the bottom.
        dropout_rate: Dropout rate in the classification head.
    
    Returns:
        VideoMAECVSClassifier model
    """
    model = VideoMAECVSClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        freeze_layers=freeze_layers,
    )
    
    return model.to(config.DEVICE)
