# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mail::SpamAssassin::Plugin::FANN is a SpamAssassin plugin that uses FANN (Fast Artificial Neural Network) for email spam/ham classification. It builds TF-IDF feature vectors from message text and optionally includes SpamAssassin rule hits as binary features.

## Build & Test Commands

```bash
perl Makefile.PL && make          # Build
make test                          # Run all tests
perl -Ilib t/fann.t                # Run single test directly
```

Prerequisites: Perl 5.14+, Mail::SpamAssassin >= 4.0, AI::FANN, Storable.

Training command:
```bash
perl -T -Ilib bin/sa-fann-train \
  --spam /var/lib/mxguardian/corpus/split/train/spam \
  --ham /var/lib/mxguardian/corpus/split/train/ham \
  --skip-rbl-checks --jobs 8 --progress
```

Always use `--skip-rbl-checks` (allows SPF/DMARC/DKIM but skips RBL checks) and `--jobs 8` to speed things up.

## Architecture

**Core plugin** (`lib/Mail/SpamAssassin/Plugin/FANN.pm`): Extends `Mail::SpamAssassin::Plugin`. Registers the `check_fann` eval rule used in SpamAssassin config. At parse time, loads a pre-trained FANN model and vocabulary from `fann_data_dir`. At scan time, tokenizes messages, computes TF-IDF vectors, appends binary rule features, runs the neural network, and caches the prediction.

**Training pipeline** (`bin/sa-fann-train`): Three-phase standalone script:
1. Tokenize all spam/ham messages and collect SA rule hits (supports multi-process via `--jobs`)
2. Prune vocabulary using chi-squared feature selection, cap text terms to `fann_vocab_cap` (rule features exempt from cap)
3. Train FANN network on TF-IDF + rule-hit vectors, save model and vocabulary (Storable format)

**Key public methods in FANN.pm**: `tokenize_text`, `tokenize_filename`, `extract_tokens`, `compute_tfidf_vector` — these are called by both the plugin and the training script.

**Feature prefixes**: Tokens are namespaced by source: `body:`, `subj:`, `from:`, `attach:`, `link:`, `utld:`, `ftld:`, `cc:`, `rule:`.

## Test Structure

Single test file `t/fann.t` with 3 tests: trains a model on 10 spam + 10 ham test messages (in `t/data/`), then verifies FANN_SPAM fires on spam and FANN_HAM fires on ham. Test config in `t/config/` uses relaxed thresholds (`fann_min_spam_count 0`, `fann_min_ham_count 0`).

## Training Corpus

Preclassified messages in Maildir format (one message per file):
- Full corpus: `/var/lib/mxguardian/corpus/{spam,ham}`
- 80/20 split: `/var/lib/mxguardian/corpus/split/{train,test}/{spam,ham}`
- FANN data files (model + vocabulary): `/var/lib/mxguardian/fann_data`

## SpamAssassin Environment

- Installed system-wide: `/usr/local/share/perl/5.36.0/Mail/SpamAssassin.pm`
- Site config directory: `/etc/mail/spamassassin`
- Using the ExtractText plugin to extract text from PDFs and image files

## Coding Conventions

- Perl strict/warnings throughout; scripts use taint mode (`perl -T`) with `untaint_file_path()`
- Debug logging via `dbg("FANN: ...")` and `info("FANN: ...")`
- Data persistence via `Storable::store`/`retrieve`
- Config parameters registered through SpamAssassin's `$conf->{parser}->register_commands()` pattern
- Scripts use `Pod::Usage` for `--help` output
