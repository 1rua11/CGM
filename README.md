## Enhanced Online CAM: Single-Stage Weakly Supervised Semantic Segmentation via Collaborative Guidance

[[arXiv]](https://arxiv.org/abs/2203.02664) [[Project]](https://rulixiang.github.io/afa) [[Poster]]()

<div align="center">

<br>
  <img width="100%" alt="AFA flowchart" src="./docs/assets/imgs/afa.png">
</div>

## Abastract

> Weakly supervised semantic segmentation with image-level annotations usually adopts multi-stage approaches, where high-quality offline CAM is generated as pseudo labels for further training, leading to a complex training process. Instead, current single-stage approaches, directly learning to segment objects with online CAM from image-level supervision, is more elegant. The quality of CAM critically determines the final segmentation performance. However, how to generate high-quality online CAM is not deeply studied in existing single-stage methods. In this paper, we propose a new single-stage framework to mine more relative target features for enhanced online CAM. Specifically, we design a novel Collaborative Guidance Mechanism, where a prior guidance block uses the original CAM to produce class-specific feature representations, improving the quality of online CAM. But such a prior is sensitive to discriminative regions of objects. Thus, we further propose a prior fusion block, in which the online segmentation prediction and the original CAM are fused to strengthen the prior guidance. Extensive experiments show that our approach achieves new state-of-the-art performance on both PASCAL VOC 2012 and MS COCO 2014 datasets, outperforming recent single-stage methods by a clear margin. The code will be released.

## Acknowledgement
