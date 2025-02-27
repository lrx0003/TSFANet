# TSFANet

  Recent advances in deep learning have led to the widespread use of convolutional neural networks (CNNs), Transformer models, and Mamba models in optical remote sensing image (ORSI) analysis, particularly for salient object detection (SOD) tasks in applications such as disaster warning, urban planning, and military surveillance. While existing methods have improved detection accuracy by optimizing feature extraction and attention mechanisms, they still face challenges when dealing with the inherent complexities of ORSI. These challenges primarily involve complex backgrounds, extreme scale variations, and topological irregularities, which significantly impact detection performance. The deeper underlying issue is the effective alignment and integration of local detail features with global semantic information. To address these challenges, we propose the Trans-Mamba Hybrid Network with Semantic Feature Alignment (TSFANet), a novel architecture that leverages the intrinsic correlations between semantic information and detail features. Our network consists of three key components: 1) the TransMamba Semantic-Detail Dual-Stream Collaborative Module (TSDSM), which combines CNN-Transformer and CNN-Mamba in a hybrid dual-branch encoder to capture both global context and multi-scale local features; 2) the Adaptive Semantic Correlation Refinement Module (ASCRM), which utilizes semantic-detail feature correlations for guided feature optimization; and 3) the Semantic-Guided Adjacent Feature Fusion Module (SGAFF), which aligns and refines multi-scale semantic features. Extensive experiments on three public RSI-SOD datasets demonstrate that our method consistently outperforms 26 state-of-the-art approaches.
