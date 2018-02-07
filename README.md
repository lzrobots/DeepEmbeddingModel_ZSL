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




| Model      |   AwA1 T1    |    u    |    s    |    H    |   CUB1 T1    |    u    |    s    |    H    |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|
| DAP [2]      |   44.1  |   0.0   |   **88.7**  |   0.0   |   40.0  |   1.7   |   67.9  |   3.3   |
| CONSE [3]     |   45.6  |   0.4   |   88.6  |   0.8   |   34.3  |   1.6   |   **72.2**  |   3.1   |
| SSE [4]       |   60.1  |   7.0   |   80.5  |   12.9  |   43.9  |   8.5   |   46.9  |   14.4  |
| DEVISE [5]    |   54.2  |   13.4  |   68.7  |   22.4  |   52.0  |   **23.8**  |   53.0  |   32.8  |
| SJE [6]       |   65.6  |   11.3  |   74.6  |   19.6  |   53.9  |   23.5  |   59.2  |   33.6  |
| LATEM [7]     |   55.1  |   7.3   |   71.7  |   13.3  |   49.3  |   15.2  |   57.3  |   24.0  |
| ESZSL [8]     |   58.2  |   6.6   |   75.6  |   12.1  |   53.9  |   12.6  |   63.8  |   21.0  |
| ALE [9]       |   59.9  |   16.8  |   76.1  |   27.5  |   54.9  |   23.7  |   62.8  |   **34.4**  |
| SYNC [10]      |   54.0  |   8.9   |   87.3  |   16.2  |   **55.6**  |   11.5  |   70.9  |   19.8  |
| SAE [11]       |   53.0  |   1.8   |   77.1  |   3.5   |   33.3  |   7.8   |   54.0  |   13.6  |
| **DEM (OURS)** | **68.44** | **32.82** | 84.71  |  **47.31** | 51.71  |   19.55  |  57.85  |  29.22  |




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
- [2] [Attribute-Based Classification forZero-Shot Visual Object Categorization](https://cvml.ist.ac.at/papers/lampert-pami2013.pdf).
  Christoph H. Lampert, Hannes Nickisch and Stefan Harmeling.
  PAMI, 2014.
- [3] [Zero-Shot Learning by Convex Combination of Semantic Embeddings](https://arxiv.org/abs/1312.5650).
  Mohammad Norouzi, Tomas Mikolov, Samy Bengio, Yoram Singer, Jonathon Shlens, Andrea Frome, Greg S. Corrado, Jeffrey Dean.
  arXiv, 2013.
- [4] [Zero-Shot Learning via Semantic Similarity Embedding](https://arxiv.org/abs/1509.04767).
  Ziming Zhang, Venkatesh Saligrama.
  ICCV, 2015.
- [5] [DeViSE: A Deep Visual-Semantic Embedding Model](http://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf).
  Andrea Frome*, Greg S. Corrado*, Jonathon Shlens*, Samy BengioJeffrey Dean, Marc’Aurelio Ranzato, Tomas Mikolov.
  NIPS, 2013.
- [6] [Evaluation of Output Embeddings for Fine-Grained Image Classification](https://arxiv.org/abs/1409.8403).
  Zeynep Akata, Scott Reed, Daniel Walter, Honglak Lee, Bernt Schiele.
  CVPR, 2015.
- [7] [Latent Embeddings for Zero-shot Classification](https://arxiv.org/abs/1603.08895).
  Yongqin Xian, Zeynep Akata, Gaurav Sharma, Quynh Nguyen, Matthias Hein, Bernt Schiele
  CVPR, 2016.
- [8] [An embarrassingly simple approach to zero-shot learning](http://proceedings.mlr.press/v37/romera-paredes15.pdf).
  Bernardino Romera-Paredes, Philip H. S. Torr.
  ICML, 2015.
- [9] [Label-Embedding for Image Classification](https://arxiv.org/abs/1503.08677).
  Zeynep Akata, Florent Perronnin, Zaid Harchaoui, Cordelia Schmid.
  PAMI, 2016.
- [10] [Synthesized Classifiers for Zero-Shot Learning](https://arxiv.org/abs/1603.00550).
  Soravit Changpinyo, Wei-Lun Chao, Boqing Gong, Fei Sha.
  CVPR, 2016.
- [11] [Semantic Autoencoder for Zero-Shot Learning](https://arxiv.org/abs/1704.08345).
  Elyor Kodirov, Tao Xiang, Shaogang Gong.
  CVPR, 2017.
