# OnGEBD: Online Generic Event Boundary Detection

This is a novel, highly innovative method added to the `step_segment` pipeline. Unlike **DDM-Net**, **EfficientGEBD**, and **DiffGEBD** (which operate offline and look at both past and future frames), **OnGEBD** works purely causally—meaning it can detect boundaries in **real-time streaming video** without any future-frame latency.

## Why is this better?
1. **Real-Time Deployment:** In a real-world factory setting, you need to know *immediately* when a sewing step finishes, not 5 seconds later.
2. **Cognitively Plausible:** Based on the Event Segmentation Theory (EST), it mimics how humans perceive boundaries—we anticipate what will happen next, and when our anticipation fails (a spike in prediction error), we register it as a new event boundary.
3. **Speed:** It requires only one forward pass per frame using a fast Recurrent Anticipator, avoiding the costly 16-step diffusion process of DiffGEBD.

## Architecture (`model.py`)
- **FeatureExtractor:** ResNet-50 spatial feature backbone.
- **ConsistentEventAnticipator (CEA):** A Causal GRU that observes frames $x_0 \dots x_{t-1}$ and predicts the representation of frame $x_t$.
- **OnlineBoundaryDiscriminator (OBD):** An MLP that compares the predicted frame $\hat{x}_t$ against the actual frame $x_t$. The prediction error (L2 norm) combined with the hidden temporal context triggers a boundary probability prediction.

## Training Concept
The model minimizes two losses simultaneously:
1. **Anticipation Loss (MSE):** Forces the CEA to accurately predict the next frame feature based on the past.
2. **Boundary Classification Loss (BCE):** Forces the OBD to correctly classify ground-truth boundaries based on the prediction error.
