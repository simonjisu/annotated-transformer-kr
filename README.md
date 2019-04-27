# Annotated-Transformer-KR

The repository that implementated of Transformer with PyTorch. Also some annotation with detail codes.

* Korean Blog: [Annotated Transformer KR](https://www.notion.so/simonjisu/Attention-Is-All-You-Need-5944fbf370ab46b091eeb64453ac3af5)
* Jupyter Notebook: 

## Getting Started

```
$ python main.py --help
```

A quick-start training is also ready for you. You can check basic settings in `run-main.sh`

```
$ sh run-main.sh
```

### Performance: Training Multi30k Result

Trained 35 Steps using NVIDIA GTX 1080 ti, Training excution time with validation: 0 h 43 m 9.6019 s

![](figs/perplexity-acc.png)

Check `Transformer.ipynb` to see translate a sample sentence

### Requirements

```
python >= 3.6
pytorch >= 1.0.0
torchtext
numpy
```

## TODO

1. Train bigger datas and make a demo server
2. Beam Search
3. Calculate BLEU Scores for Translation Task

## references

I checked a lot of references. Please visit them and learn it!

* paper : https://arxiv.org/abs/1706.03762
* reference blog: http://nlp.seas.harvard.edu/2018/04/03/attention.html
* reference code: https://github.com/jadore801120/attention-is-all-you-need-pytorch