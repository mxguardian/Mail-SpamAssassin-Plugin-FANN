# Mail::SpamAssassin::Plugin::FANN

A SpamAssassin plugin that uses [FANN](http://leenissen.dk/fann/wp/) (Fast Artificial Neural Network) to classify email as spam or ham.

The plugin builds TF-IDF feature vectors from message text (body, subject, from name, attachment filenames) and optionally includes binary SpamAssassin rule hit features. A trained FANN model produces a prediction score between 0 (ham) and 1 (spam), which is matched against configurable threshold ranges to fire rules.

## Installation

```bash
perl Makefile.PL
make
make test
make install
```

### Prerequisites

- Perl 5.14+
- [Mail::SpamAssassin](https://metacpan.org/pod/Mail::SpamAssassin) 4.0+
- [AI::FANN](https://metacpan.org/pod/AI::FANN)

## Configuration

Add to your SpamAssassin config (e.g. `local.cf`):

```
loadplugin Mail::SpamAssassin::Plugin::FANN

fann_data_dir       /var/lib/spamassassin/fann

body    FANN_SPAM   eval:check_fann(0.50, 1.00)
score   FANN_SPAM   2.0

body    FANN_HAM    eval:check_fann(0.00, 0.50)
score   FANN_HAM    -2.0
```

The `check_fann(low, high)` eval rule fires when the prediction falls within `[low, high]`. You can define as many rules as you like with different threshold ranges:

```
body    FANN_SPAM_HI    eval:check_fann(0.85, 1.00)
score   FANN_SPAM_HI    3.0

body    FANN_SPAM_LO    eval:check_fann(0.50, 0.85)
score   FANN_SPAM_LO    1.5

body    FANN_HAM_HI     eval:check_fann(0.00, 0.15)
score   FANN_HAM_HI     -3.0

body    FANN_HAM_LO     eval:check_fann(0.15, 0.50)
score   FANN_HAM_LO     -1.5
```

### Configuration Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `fann_data_dir` | (none) | Directory for model and vocabulary files |
| `fann_min_word_len` | 4 | Minimum token length |
| `fann_max_word_len` | 24 | Maximum token length |
| `fann_vocab_cap` | 10000 | Max vocabulary terms (ranked by chi-squared) |
| `fann_min_spam_count` | 100 | Min spam messages required to enable prediction |
| `fann_min_ham_count` | 100 | Min ham messages required to enable prediction |
| `fann_learning_rate` | 0.1 | FANN learning rate |
| `fann_momentum` | 0.1 | FANN training momentum |
| `fann_train_epochs` | 50 | Number of training epochs |
| `fann_train_algorithm` | FANN_TRAIN_RPROP | Training algorithm |
| `fann_stopwords` | (none) | Space-separated stopwords to ignore (can be specified multiple times) |

## Training

Train a model using `sa-fann-train`:

```bash
sa-fann-train --spam /path/to/spam --ham /path/to/ham
```

Messages should be one file per message. The tool:

1. Tokenizes each message body, subject, from name, and attachment filenames
2. Builds a vocabulary ranked by chi-squared informativeness
3. Optionally runs each message through SpamAssassin to collect rule hits as binary features
4. Trains a FANN neural network and saves the model to `fann_data_dir`

### Training Options

```
--spam path         Directory containing spam messages (required)
--ham path          Directory containing ham messages (required)
--epochs N          Override fann_train_epochs
--vocab-cap N       Override fann_vocab_cap
--no-rules          Text-only model (skip rule feature collection)
--hidden N          Number of hidden neurons (default: sqrt of input size)
--progress          Show progress counter
-C path             SpamAssassin config directory
--siteconfigpath    Site config directory
-p file             User preferences file
-L                  Local tests only (no network tests)
-D [areas]          Debug output
```

## Utilities

### sa-fann-dump

Inspect the trained vocabulary:

```bash
sa-fann-dump --path /var/lib/spamassassin/fann/vocabulary.data
sa-fann-dump --prefix body:    # show only body tokens
sa-fann-dump --rules           # show rule features
```

### fann-split-corpus

Split a corpus into training and test sets:

```bash
fann-split-corpus --seed 42 /path/to/spam /path/to/ham /path/to/output
```

### fann-test-accuracy

Evaluate model accuracy against test sets:

```bash
fann-test-accuracy /path/to/test/spam /path/to/test/ham
fann-test-accuracy -m /path/to/test/spam /path/to/test/ham  # show misclassified
```

## License

Copyright 2026 MXGuardian. Licensed under the Apache License, Version 2.0.
