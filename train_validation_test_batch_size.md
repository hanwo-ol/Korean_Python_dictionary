테스트, 검증 셋에서 배치사이즈 크기

링크 1
"https://stackoverflow.com/questions/54413160/training-validation-testing-batch-size-ratio"   
```

You can follow the advice from the other answers for the dataset split ratio.
However, the batch size has absolutely nothing to do with how you've split your datasets.
The batch size determines how many training examples are processed in parallel for training/inference.
The batch size at training time can affect how fast and how well your training converges.
You can find a discussion of this effect here.   ---> 아래 링크 2
Thus, for train_batch_size, it's worth picking a batch size that is neither too small nor too large (as discussed in the previously linked discussion).
For some applications, using the largest possible training batches can actually be desirable, but in general, you select it through experiments and validation.
However, for validation_batch_size and test_batch_size, you should pick the largest batch size that your hardware can handle without running out of memory and crashing.
Finding this is usually a simple trial and error process.
The larger your batch size at inference time, the faster it will be, since more inputs can be processed in parallel.
EDIT: Here's an additional useful link (Pg. 276) for the training batch size trade-off from Goodfellow et al's deep learning book.
```

링크 2   
"https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu"

링크 3 DEEP LEARNING BOOK   
"https://www.deeplearningbook.org/contents/optimization.html"
