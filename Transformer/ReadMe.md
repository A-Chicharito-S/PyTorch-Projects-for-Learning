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
 

For an illustration of the code, please refer to my blog [[here]]()
