# Transformer

This is the implementation of the **Transformer** model from "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)".

The following projects/websites are used as code references:

- [SDISS](https://github.com/L-Zhe/SDISS) is a Transformer-based sentence simplification model
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) is a post about the Transformer and its implementation
- [How to code The Transformer in Pytorch](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec) is a post about the Transformer and its implementation

The implementation of the **Transformer** model is in the '**model**' folder and inside the '**toolkit**' folder I have the optimizer, loss, and the framework responsible for training implemented.

To run the code, please simply run '**test.py**' (where like [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) a simple re-construction task is performed)

The sequence reconstruction result for [ 1  5 10  5  4  3  6  0  0  2] is: 
> greedy search: [(1), 5, 10, 4, 5, 6, 3, 5, (2)]
> 
> beam search: [(1), 5, 10, 5, 4, 6, 3, 5, (2)]
The model configration is the same with that of [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) however did not get a good result, I suspect the reasons are two-fold:
1. my implemention used the paddings (index set to 0) and the implementation from [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) did not use paddings when training the re-construction task.
2. my implementation used the BOS (index set to 1) and EOS (index set to 2) tokens for each sentence and [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) only used the BOS token; thus it will be harder for the Transformer model of my version to learn the dependency with the same amount of training data. (like [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html), the model is trained for 10 epochs with a batch size of 30 and 20 batches per epoch) 

For an illustration of the code, please refer to my blog [[here]](https://a-chicharito-s.github.io/coding%20skill/2022/02/10/transformer-implementation/)
