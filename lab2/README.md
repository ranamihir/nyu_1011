In order:

* [pytorch_basics](pytorch_basics.ipynb)
* [logistic_regression](logistic_regression.ipynb)
* [datasets_and_dl_training](datasets_and_dl_training.ipynb)


## Top 10 Things to Check When Your Code Doesn't Seem to be Working Right

(...that's unrelated to CUDA/cuDNN issues - that deserves its own Medium post.)

1. You did not call `.zero_grad()` on your model/optimizer every iteration.
2. You did not call `.backward` after computing your loss.
3. Using `NLLLoss()` and `CrossEntropyLoss()` wrongly (`NLLLoss` works on log-probabilities, `CrossEntropyLoss` works on law logits).
    * Note that you can either to either and still have your model sort of converge, even though your loss is computed wrongly!
4. Your data / Tensors have the wrong dimension, and every variant of this problem.
5. You computed your loss with the wrong sign. Particularly bad when you have e.g. multiple regularization losses, and just one of them have the wrong sign.
6. You forgot to call `.train()` and `.eval()` when training your model / computing validation/test losses
7. You forgot to call `.zero_grad()`, and now you're out of memory.
8. You carelessly slice your tensors, and gradients either do no propagate, or you're working on the wrong slice altogether.
9. You apply softmax, or any other reductive function over the wrong dimension.
10. You called `.squeeze()` without specifying a dimension, and your code breaks when you try a batch size of 1.

## Other Pro-Tips
1. When testing/debugging your model, **overfit to the smallest amount of data possible**.
    * Turn off all regularization, let your model overfit to 1, then 2, then 4 data points, and so on. If your model can't overfit (and deep learning models should overfit easily), something is *very* wrong.
2. Clip gradients.
    * This is magic that sometimes helps RNNs. See: exploding gradients.
3. Finish 1 Epoch before letting your code run while you go to bed.
    * There is a decent chance that your code breaks on the last batch (which may be less than your desired batch size), or on logging / evaluation / saving. Run on a small fraction of data if necessary.