dataset: ./dataset_tas
methods:
1, ms-tcn:
+ paper: https://arxiv.org/pdf/1903.01945
+ source code: https://github.com/yabufarha/ms-tcn 
===
2, FACT:
+ paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.pdf
+ source code: https://github.com/ZijiaLewisLu/CVPR2024-FACT
===
3, TQT:
+ paper: https://github.com/tqosu/TQT
+ source code: https://openaccess.thecvf.com/content/WACV2026/papers/Wang_Timestamp_Query_Transformer_for_Temporal_Action_Segmentation_WACV_2026_paper.pdf
===

Implement/Setup this method in folder  ./src/TAS
Following steps for each method:
1. clone paper, source code
2. check data training format => whether align with my dataset at ./dataset_tas => if not ? can process ?
3. write script to training (i'll run by my-self)
