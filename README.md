# dialogue-transformer-e2e

This is a repository for End-to-end Dialogue Transformer project for [Statistical Dialogue Systems](http://ufal.mff.cuni.cz/courses/npfl099) course.

## Goals
- [x] Improve sequicity comments
- [ ] ~Use PyTorch's `nn.transformer` to implement Sequicity style dialogue system~
    - [x] Try to run Sequicity as is - this *should* be quite easy.
    - [ ] ~Rewrite classes `SimpleDynamicEncoder`, `BSpanDecoder`, and `ResponseDecoder` from `tsd_net.py` to use transformer instead of RNNs. This will probably involve also adjusting `TSD` class.~
- [x] Compare it with existing dialogue systems (probably Sequicity, mainly)
- [ ] ~Improve performance by utilizing pre-trained LM.~
- [x] Implement it in tensorflow

## Results

| System                | F1 success | BLEU      |
|-----------------------|------------|-----------|
| Transformer           | 0.770      | **0.327** |
| Transformer \ copynet | 0.710      | 0.315     |
| Sequicity             | **0.854**  | 0.253     |

We have shown that transformer with copy mechanism comparable performance with Sequicity.
We believe the system could be improved by utilizing a pre-trained language model (BERT, GPT-{2|3}, MASS, XLNet, ...)

Although the the success F1 score did not superseded our baseline, our model has BLEU score of responses 7.4% larger than Sequicity. 
We think that the worse performance of Transformer, compared to recurrent neural networks may be caused by the small amount of data we have, relatively low batch size and generally lower stability of training ([Training Tips for the Transfomer Model](https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf)).

## Papers
Papers related to this work

- [Sequicity](https://www.comp.nus.edu.sg/~kanmy/papers/acl18-sequicity.pdf)
- [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/pdf/1603.06393.pdf) - the copy mechanism referenced from Sequicity, quite an interesting paper
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) - the transformer architecture
- [Hello, It's GPT-2](https://arxiv.org/pdf/1907.05774.pdf)
- [ALBERT: A Lite BERT](https://arxiv.org/pdf/1909.11942.pdf) - IHMO (ondrej) the methods described in this paper might be easier to use with limited computational resources compared to other pretrained transformers (BERT, GPT-2, XLNet, Transformer-XL, ...)
- [Training Tips for the Transfomer Model](https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf) - A nice paper form UFAL about practical tips for training transformer, might be useful
- [On Layer Normalization in the Transformer Architecture](https://openreview.net/forum?id=B1x8anVFPr) - They stabilize the training by placing layer normalization inside the residual block and before the multi-head attention (Pre-LN). Therefore they can remove warm-up and use a larger learning rate.

## Acknowledgements
- The transformer is the official [Tensorflow implementation](https://github.com/tensorflow/models/tree/master/official/transformer).
- Sequicity implementation from the [authors' repository](https://github.com/WING-NUS/sequicity)
