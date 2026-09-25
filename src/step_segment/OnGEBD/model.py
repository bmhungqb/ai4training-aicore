import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FeatureExtractor(nn.Module):
    """Extracts spatial features from frames using ResNet50."""
    def __init__(self, feature_dim=2048, freeze_backbone=False):
        super().__init__()
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            # Fallback for older torchvision versions
            resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = feature_dim
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.backbone(x) # (B*T, 2048, 1, 1)
        feat = feat.view(B, T, self.feature_dim)
        return feat

class ConsistentEventAnticipator(nn.Module):
    """
    Predicts the next frame's feature based on past frames.
    Uses a Causal GRU to ensure zero look-ahead (Online requirement).
    """
    def __init__(self, feature_dim=2048, hidden_dim=512):
        super().__init__()
        self.rnn = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.predictor = nn.Linear(hidden_dim, feature_dim)
        
    def forward(self, x):
        # x shape: (B, T, feature_dim)
        # rnn_out: (B, T, hidden_dim)
        rnn_out, _ = self.rnn(x)
        # Predict the next frame's feature for each timestep
        pred_next_x = self.predictor(rnn_out)
        return pred_next_x, rnn_out

class OnlineBoundaryDiscriminator(nn.Module):
    """
    Detects boundaries by comparing the predicted next frame and the actual next frame.
    Spikes in prediction error indicate an event boundary.
    """
    def __init__(self, feature_dim=2048, hidden_dim=512):
        super().__init__()
        # Input: [actual_x, predicted_x, error_norm, rnn_hidden]
        input_dim = feature_dim * 2 + 1 + hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Binary boundary prediction (logits)
        )
        
    def forward(self, actual_x, pred_x, rnn_hidden):
        """
        actual_x: (B, T, feature_dim)
        pred_x: (B, T, feature_dim)
        rnn_hidden: (B, T, hidden_dim)
        """
        # Calculate L2 error norm between actual and predicted
        error_norm = torch.norm(actual_x - pred_x, p=2, dim=-1, keepdim=True) # (B, T, 1)
        
        # Concatenate all cues
        cues = torch.cat([actual_x, pred_x, error_norm, rnn_hidden], dim=-1) # (B, T, input_dim)
        
        # Predict boundary probability logits
        logits = self.mlp(cues) # (B, T, 1)
        return logits.squeeze(-1)

class OnGEBDModel(nn.Module):
    """
    Full Online Generic Event Boundary Detection Model.
    Mimics human event perception: anticipates the future and flags boundaries on surprise.
    """
    def __init__(self, feature_dim=2048, hidden_dim=512, freeze_backbone=False):
        super().__init__()
        self.feature_extractor = FeatureExtractor(feature_dim, freeze_backbone)
        self.anticipator = ConsistentEventAnticipator(feature_dim, hidden_dim)
        self.discriminator = OnlineBoundaryDiscriminator(feature_dim, hidden_dim)
        
    def forward(self, x):
        """
        x: video frames of shape (B, T, C, H, W)
        """
        # 1. Extract spatial features for all frames
        features = self.feature_extractor(x) # (B, T, 2048)
        
        # 2. Anticipate next frames
        # For a causal model, prediction at time t corresponds to predicting frame t+1.
        # We shift the input to the anticipator so that it only sees up to t-1 when predicting t.
        
        # Create a zero-padded dummy frame for the start
        B, T, C = features.shape
        dummy = torch.zeros(B, 1, C, device=features.device)
        past_features = torch.cat([dummy, features[:, :-1, :]], dim=1) # (B, T, 2048)
        
        pred_features, rnn_hidden = self.anticipator(past_features) # (B, T, 2048), (B, T, 512)
        
        # 3. Discriminate boundaries based on anticipation error
        boundary_logits = self.discriminator(features, pred_features, rnn_hidden) # (B, T)
        
        # Return logits and the anticipation loss for auxiliary training
        # Anticipation loss forces the model to actually learn to predict the next frame
        anticipation_loss = F.mse_loss(pred_features, features)
        
        return boundary_logits, anticipation_loss

    def init_state(self, batch_size=1, device="cpu"):
        """
        Initialize streaming state for true frame-by-frame online inference.
        Call once at the start of a video stream.
        """
        return {
            "prev_feature": torch.zeros(batch_size, self.feature_extractor.feature_dim, device=device),
            "gru_hidden": None,  # nn.GRU will default-init zeros internally
        }

    @torch.no_grad()
    def step(self, frame, state):
        """
        Single-frame causal inference (the O(1)/frame, zero-look-ahead path
        the README describes, but that `forward()` alone does not expose).

        frame: (B, C, H, W) - current frame x_t
        state: dict returned by init_state() / a previous step() call

        Returns:
            boundary_prob: (B,) sigmoid probability that x_t is a boundary
            new_state: dict to pass into the next step() call
        """
        was_training = self.training
        self.eval()

        feat = self.feature_extractor(frame.unsqueeze(1)).squeeze(1)  # (B, feature_dim)

        # Anticipate x_t from the state accumulated over x_0..x_{t-1} only.
        rnn_input = state["prev_feature"].unsqueeze(1)  # (B, 1, feature_dim)
        rnn_out, new_hidden = self.anticipator.rnn(rnn_input, state["gru_hidden"])
        pred_feat = self.anticipator.predictor(rnn_out.squeeze(1))  # (B, feature_dim)

        logits = self.discriminator(feat.unsqueeze(1), pred_feat.unsqueeze(1), rnn_out).squeeze(1)  # (B,)
        boundary_prob = torch.sigmoid(logits)

        new_state = {"prev_feature": feat, "gru_hidden": new_hidden}
        self.train(was_training)
        return boundary_prob, new_state
