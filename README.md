# Language Translation using Handwritten Text Recognition

## Datasets
- Dataset for German-English pairs - https://www.statmt.org/wmt19/translation-task.html 
- Dataset for English-Kannada pairs - https://www.kaggle.com/datasets/parvmodi/english-to-kannada-machine-translation-dataset 
- IAM Handwriting Database - https://fki.tic.heia-fr.ch/databases/iam-handwriting-database 

## Machine Translation
### Preprocessing
#### Tokenization and Vocabulary Building
- Tokenizer used - WordPiece tokenizer from the `keras_nlp` library.
- Vocabulary size - 15000  

For more information on how the tokenizer works, refer to this [guide](https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt)

#### Creating the Dataset
1. The dataset is created using the `tf.data.Dataset` API.
2. The primary language input is padded to the maximum length of the input sequence.
3. Start and end tokens are added to the target language input and padded to the maximum length of the target sequence.
4. The dataset is then batched and cached for performance.

### Model
- The `TokenAndPositionEmbedding` layer of the `keras_nlp` library is used to create an embedding vector for each token in the input sequence along with creating a positional embedding vector which encodes the position of each token in the sequence.
- The model uses a sequence-to-sequence transformer architecture with one encoder and one decoder.
- The source sequence is passed through `TransformerEncoder` layer which produces an a new representation of the sequence. This is passed to the `TransformerDecoder` layer along with the target sequence to produce the next token in the sequence.

The Transformer architecture is the same as the one described in the paper - [Attention is all you need](https://arxiv.org/abs/1706.03762)

### Training
- The model is trained for 20 epochs with a batch size of 256.
- The Adam optimizer is used with default learning rate.
- The loss function used is the SparseCategoricalCrossentropy loss function.
- The BLEU score is used to evaluate the model.

BLEU score is a metric used to evaluate the quality of a machine translated text. It is based on the n-gram precision of the translated text with respect to the reference text. To know more about BLEU score, refer to this [guide](https://towardsdatascience.com/foundations-of-nlp-explained-bleu-score-and-wer-metrics-1a5ba06d812b)

### Results
The model achieves a BLEU score of 0.1 on the test set (en-kn), even after using regularization methods such as dropouts.
Reasons could be:
- The model takes a long time to train. The model was trained for 20 epochs which took around 5 hours on a GPU.
- The model might not be complex enough to learn the patterns in the data. The model has only one encoder and one decoder layer. The model could be made more complex by adding more layers.

## Handwritten Text Recognition
### Preprocessing
#### Creating the Dataset
1. Data is cleaned and stored in python lists. One for image paths and the other for corresponding labels.
2. Maximum length of the labels is calculated and used to pad the labels along with the character vocabulary size.
3. The character vocabulary is created using the `StringLookup` layer of the `keras` library. To know more about the layer, refer to the [documentation](https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup/).
4. Images are resized to 128x32 and converted to grayscale. The aspect ratio is maintained.
5. The dataset is then batched and cached for performance. 

### Model
- The model uses two convolutional layers followed by a dense layer and two bidirectional LSTM layers.
- The model uses the CTC loss function to train the model. The CTC loss function is used to train models for sequence prediction problems. To know more about the CTC loss function, refer to this [guide](https://distill.pub/2017/ctc/).

### Training
- The model is trained for 90 epochs with a batch size of 64.
- The Adam optimizer is used with default learning rate.
- The loss function used is the CTC loss function.
- The edit distance is used to evaluate the model.

The edit distance is a metric used to evaluate the quality of a machine translated text. It is based on the number of insertions, deletions and substitutions required to convert the predicted text to the actual text. To know more about edit distance, refer to this [article](https://en.wikipedia.org/wiki/Edit_distance)


