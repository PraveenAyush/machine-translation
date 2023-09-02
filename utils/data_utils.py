import numpy as np
import tensorflow as tf
import keras_nlp as kn
import pandas as pd

from sklearn.model_selection import train_test_split

def create_dataset(primary_lang_path, secondary_lang_path, num_samples, output_dir):
    """
    Create a dataset from the given data

    Args:
        primary_lang_path: Path to the primary language data
        secondary_lang_path: Path to the secondary language data
        num_samples: Number of samples to be used
        output_dir: Path to the output directory

    Returns:
        None
    """
    # Read the data
    primary_lang = pd.read_table(primary_lang_path, header=None, encoding='utf-8')
    secondary_lang = pd.read_table(secondary_lang_path, header=None, encoding='utf-8')
    data = pd.concat([primary_lang, secondary_lang], axis=1)

    # Shuffle the data
    data = data.sample(n=num_samples, random_state=42)

    data.to_csv(output_dir + '/data.tsv', sep='\t', index=False, header=False)
        

def split_data(data_path, output_dir, val_size=0.2, test_size=0.33):
    """
    Split the data into training, validation and test sets

    Args:
        data: Path to tsv file
        val_size: Size of the total validation data including test set
        test_size: Size of the test set as a part of the validation data

    Returns:
        None
    """
    data = pd.read_csv(data_path, sep='\t', header=None, on_bad_lines='skip', names=['de', 'en'])
    data.dropna(inplace=True)

    # Split the data into training, validation and test sets
    train_pairs, temp_pairs = train_test_split(data, test_size=val_size, random_state=42)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=test_size, random_state=42)

    # Save the data
    train_pairs.to_csv(output_dir + '/train.tsv', sep='\t', index=False, header=False)
    val_pairs.to_csv(output_dir + '/val.tsv', sep='\t', index=False, header=False)
    test_pairs.to_csv(output_dir + '/test.tsv', sep='\t', index=False, header=False)


def compute_vocabulary(samples, vocab_size, reserved_tokens, vocab_file_path):
    """
    Create a vocabulary from the given samples to feed into the WordPiece Tokenizer

    Args:
        samples: List of samples
        vocab_size: Size of the vocabulary
        reserved_tokens: List of reserved tokens

    Returns:
        Vocabulary
    """

    ds = tf.data.Dataset.from_tensor_slices(samples)
    vocab = kn.tokenizers.compute_word_piece_vocabulary(
        ds.batch(1000).prefetch(2),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
        vocabulary_output_file=vocab_file_path
    )
    return vocab


def preprocess_batch(prim_lang, sec_lang, prim_tokenizer, sec_tokenizer, max_seq_len):
    """
    Preprocess a batch of data (tokenization, padding and adding start and end tokens for decoder inputs)

    Args:
        prim_lang: Primary language data
        sec_lang: Secondary language data
        prim_tokenizer: Primary language tokenizer
        sec_tokenizer: Secondary language tokenizer
        max_seq_len: Maximum sequence length

    Returns:
        inputs: Dictionary of inputs with keys 'encoder_inputs' and 'decoder_inputs'
        targets: target sentence offset by one step
    """

    # Tokenize the languages
    prim_lang = prim_tokenizer(prim_lang)
    sec_lang = sec_tokenizer(sec_lang)

    # Pad the primary language to max_seq_len
    prim_start_end_packer = kn.layers.StartEndPacker(
        sequence_length=max_seq_len,
        pad_value=prim_tokenizer.token_to_id('[PAD]')
    )
    prim_lang = prim_start_end_packer(prim_lang)

    # Add the start and end tokens to the secondary language and pad it to max_seq_len
    sec_start_end_packer = kn.layers.StartEndPacker(
        sequence_length=max_seq_len + 1,
        start_value=sec_tokenizer.token_to_id('[START]'),
        end_value=sec_tokenizer.token_to_id('[END]'),
        pad_value=sec_tokenizer.token_to_id('[PAD]')
    )
    sec_lang = sec_start_end_packer(sec_lang)

    return ({
        'encoder_inputs': prim_lang,
        'decoder_inputs': sec_lang[:, :-1] # type: ignore
    }, sec_lang[:, 1:]) # type: ignore


def make_dataset(pairs, batch_size, prim_tokenizer, sec_tokenizer, max_seq_len):
    prim_texts, sec_texts = zip(*pairs)

    prim_texts = list(prim_texts)
    sec_texts = list(sec_texts)

    dataset = tf.data.Dataset.from_tensor_slices((prim_texts, sec_texts))
    dataset = dataset.batch(batch_size).map(lambda x, y: preprocess_batch(
        x, y, prim_tokenizer=prim_tokenizer, sec_tokenizer=sec_tokenizer, max_seq_len=max_seq_len
        ), num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.shuffle(2048).prefetch(16).cache()


def decode_sequences(input_sequences, prim_tokenizer, sec_tokenizer, max_seq_len, model):
    """
    Decode a batch of sequences

    Args:
        input_sequences: Batch of sequences

    Returns:
        Decoded sequences
    """
    batch_size = tf.shape(input_sequences)[0]

    # Tokenize the encoder inputs
    encoder_input_tokens = prim_tokenizer(input_sequences).to_tensor(shape=(None, max_seq_len))

    # Output the next token probabilities
    def next(prompt, cache, index):
        logits = model([encoder_input_tokens, prompt], training=False)[:, index-1, :]

        # Ignore the hidden states
        hidden_states = None
        return logits, hidden_states, cache
    
    # Build a prompt of length 40 with the start token and padding
    length = 40
    start = tf.fill((batch_size, 1), sec_tokenizer.token_to_id('[START]'))
    end = tf.fill((batch_size, length - 1), sec_tokenizer.token_to_id('[PAD]'))
    prompt = tf.concat([start, end], axis=1)

    # Decode the sequences
    generated_tokens = kn.samplers.GreedySampler()(
        next,
        prompt,
        end_token_id=sec_tokenizer.token_to_id('[END]'),
        index=1 # Start sampling after the start token
    )
    generated_sequences = sec_tokenizer.detokenize(generated_tokens).numpy()[0].decode('utf-8')

    return generated_sequences
