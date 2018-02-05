# DeepEmbeddingModel_ZSL
Tensorflow code for CVPR 2017 paper: [Learning a Deep Embedding Model for Zero-Shot Learning](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Learning_a_Deep_CVPR_2017_paper.pdf)

[Li Zhang](http://www.robots.ox.ac.uk/~lz/)


# Data
Download data from [here](http://www.robots.ox.ac.uk/~lz/DEM_cvpr2017/data.zip) and unzip it `unzip data.zip`.

# Run
`AwA_attribute.py` will gives you ZSL performance on AwA with attribute.

`AwA_wordvector.py` will gives you ZSL performance on AwA with wordvector.

`AwA_fusion.py` will gives you ZSL performance on AwA with attribute and wordvector fusion.

`CUB_attribute.py`will gives you ZSL performance on CUB with attribute.

# GBU setting

ZSL and GZSL performance evaluated under GBU setting [1]: ResNet feature, GBU split, averaged per class accuracy.

`AwA1_GBU.py` will gives you ZSL and GZSL performance on AwA1 with attribute under GBU setting [1].

`CUB1_GBU.py` will gives you ZSL and GZSL performance on CUB1 with attribute under GBU setting [1].

|            |                 AwA1                  |                  CUB1                 |
|            |   ZSL   |           GZSL              |   ZSL   |           GZSL              |
| Model      |   T1    |    u    |    s    |    H    |   T1    |    u    |    s    |    H    |
| DAP        |   44.1  |   0.0   |   88.7  |   0.0   |         |         |         |         |
| CONSE      |   45.6  |   0.4   |   88.6  |   0.8   |         |         |         |         |
| SSE        |   60.1  |   7.0   |   80.5  |   12.9  |         |         |         |         |
| DEVISE     |   54.2  |   13.4  |   68.7  |   22.4  |         |         |         |         |
| SJE        |   65.6  |   11.3  |   74.6  |   19.6  |         |         |         |         |
| LATEM      |   55.1  |   7.3   |   71.7  |   13.3  |         |         |         |         |
| ESZSL      |   58.2  |   6.6   |   75.6  |   12.1  |         |         |         |         |
| ALE        |   59.9  |   16.8  |   76.1  |   27.5  |         |         |         |         |
| SYNC       |   54.0  |   8.9   |   87.3  |   16.2  |         |         |         |         |
| DEM (OURS) |         |         |         |         |         |         |         |         |



## Citing

If you use this code in your research, please use the following BibTeX entry.

```
@inproceedings{zhang2017learning,
  title={Learning a deep embedding model for zero-shot learning},
  author={Zhang, Li and Xiang, Tao and Gong, Shaogang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```

## References

- [1] [Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly](https://arxiv.org/abs/1707.00600).
  Yongqin Xian, Christoph H. Lampert, Bernt Schiele, Zeynep Akata.
  arXiv, 2017.
