# <@LICENSE>
# Copyright 2026 MXGuardian
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# </@LICENSE>

=head1 NAME

Mail::SpamAssassin::Plugin::FANN - check messages using Fast Artificial Neural Network library

=head1 SYNOPSIS

  loadplugin Mail::SpamAssassin::Plugin::FANN

=head1 DESCRIPTION

This plugin checks emails using the FANN (Fast Artificial Neural Network) library.

Training is performed by the standalone C<sa-fann-train> tool, which reads
spam and ham message directories directly to build a vocabulary and train the
FANN model.  This plugin handles inference only.

=cut

package Mail::SpamAssassin::Plugin::FANN;

use strict;
use warnings;
use re 'taint';

our $VERSION = '0.11';

use AI::FANN qw(:all);
use Storable qw(store retrieve);
use File::Spec;

use Mail::SpamAssassin;
use Mail::SpamAssassin::Plugin;
use Mail::SpamAssassin::Util qw(untaint_file_path);

our @ISA = qw(Mail::SpamAssassin::Plugin);

sub dbg { my $msg = shift; Mail::SpamAssassin::Logger::dbg("FANN: $msg", @_); }
sub info { my $msg = shift; Mail::SpamAssassin::Logger::info("FANN: $msg", @_); }

sub new {
  my ($class, $mailsa) = @_;

  $class = ref($class) || $class;
  my $self = $class->SUPER::new($mailsa);
  bless ($self, $class);

  $self->set_config($mailsa->{conf});
  $self->register_eval_rule("check_fann", $Mail::SpamAssassin::Conf::TYPE_BODY_EVALS);

  return $self;
}

sub set_config {
  my ($self, $conf) = @_;
  my @cmds;

=over 4

=item fann_data_dir dirname (default: undef)

Where FANN plugin will store its data.

=item fann_min_word_len n (default: 2)

Minimum token length considered when building the vocabulary and feature vectors.

=item fann_max_word_len n (default: 24)

Maximum token length considered when building the vocabulary and feature vectors.

=item fann_vocab_cap n (default: 10000)

Maximum number of vocabulary terms to retain; terms are ranked by chi-squared
informativeness and the lowest-scoring terms are pruned when exceeded.

=item fann_min_spam_count n (default: 100)

Minimum number of spam messages in the vocabulary required to enable prediction.

=item fann_min_ham_count n (default: 100)

Minimum number of ham messages in the vocabulary required to enable prediction.

=item fann_learning_rate f (default: 0.1)

Learning rate used by the underlying FANN network during training (used by sa-fann-train).

=item fann_momentum f (default: 0.1)

Momentum used for training updates (used by sa-fann-train).

=item fann_train_epochs n (default: 50)

Number of training epochs (used by sa-fann-train).

=item fann_train_algorithm FANN_TRAIN_QUICKPROP|FANN_TRAIN_RPROP|FANN_TRAIN_BATCH|FANN_TRAIN_INCREMENTAL (default: FANN_TRAIN_RPROP)

Algorithm used by Fann neural network used when training, might increase speed depending on the data volume.

=item fann_stopwords words (default: "the and for with that this from there their have be not but you your")

Space-separated list of stopwords to ignore when tokenizing text.

=back

=cut

  push(@cmds, {
    setting => 'fann_data_dir',
    is_admin => 1,
    default => undef,
    type => $Mail::SpamAssassin::Conf::CONF_TYPE_STRING,
  });
  push(@cmds, {
    setting => 'fann_min_word_len',
    is_admin => 1,
    default => 2,
    type => $Mail::SpamAssassin::Conf::CONF_TYPE_NUMERIC,
  });
  push(@cmds, {
    setting => 'fann_max_word_len',
    is_admin => 1,
    default => 24,
    type => $Mail::SpamAssassin::Conf::CONF_TYPE_NUMERIC,
  });
  push(@cmds, {
    setting => 'fann_vocab_cap',
    is_admin => 1,
    default => 10000,
    type => $Mail::SpamAssassin::Conf::CONF_TYPE_NUMERIC,
  });
  push(@cmds, {
    setting => 'fann_min_spam_count',
    is_admin => 1,
    default => 100,
    type => $Mail::SpamAssassin::Conf::CONF_TYPE_NUMERIC,
  });
  push(@cmds, {
    setting => 'fann_min_ham_count',
    is_admin => 1,
    default => 100,
    type => $Mail::SpamAssassin::Conf::CONF_TYPE_NUMERIC,
  });
  push(@cmds, {
    setting => 'fann_learning_rate',
    is_admin => 1,
    default => 0.1,
    type => $Mail::SpamAssassin::Conf::CONF_TYPE_NUMERIC,
  });
  push(@cmds, {
    setting => 'fann_momentum',
    is_admin => 1,
    default => 0.1,
    type => $Mail::SpamAssassin::Conf::CONF_TYPE_NUMERIC,
  });
  push(@cmds, {
    setting => 'fann_train_epochs',
    is_admin => 1,
    default => 50,
    type => $Mail::SpamAssassin::Conf::CONF_TYPE_NUMERIC,
  });
  push(@cmds, {
    setting => 'fann_train_algorithm',
    is_admin => 1,
    default => FANN_TRAIN_RPROP,
    code        => sub {
        my ($self, $key, $value, $line) = @_;
	my %algorithm_map = (
            'FANN_TRAIN_QUICKPROP'    => FANN_TRAIN_QUICKPROP,
            'FANN_TRAIN_RPROP'        => FANN_TRAIN_RPROP,
            'FANN_TRAIN_BATCH'        => FANN_TRAIN_BATCH,
            'FANN_TRAIN_INCREMENTAL'  => FANN_TRAIN_INCREMENTAL,
        );
        if (!exists $algorithm_map{$value}) {
            return $Mail::SpamAssassin::Conf::INVALID_VALUE;
        }
        $self->{fann_train_algorithm} = $algorithm_map{$value};
    },
    type => $Mail::SpamAssassin::Conf::CONF_TYPE_NUMERIC,
  });
  push(@cmds, {
    setting => 'fann_stopwords',
    is_admin => 1,
    default => 'the and for with that this from there their have be not but you your',
    type => $Mail::SpamAssassin::Conf::CONF_TYPE_STRING,
  });

  $conf->{parser}->register_commands(\@cmds);
}

sub finish_parsing_end {
  my ($self, $opts) = @_;

  my $conf = $self->{main}->{conf};
  my $nn_data_dir = $conf->{fann_data_dir};

  return unless defined $nn_data_dir;

  $nn_data_dir = Mail::SpamAssassin::Util::untaint_file_path($nn_data_dir);
  if (not -d $nn_data_dir) {
    dbg("fann_data_dir is invalid");
    return;
  }

  # Load FANN model
  my $dataset_path = File::Spec->catfile($nn_data_dir, 'fann.model');
  if (-f $dataset_path) {
    eval {
      $self->{neural_model} = AI::FANN->new_from_file($dataset_path);
      1;
    } or do {
      my $err = $@ || 'unknown';
      info("Failed to load neural model from $dataset_path: $err");
    };
  }

  # Load vocabulary and precompute index/IDF
  my $vocab_path = File::Spec->catfile($nn_data_dir, 'vocabulary.data');
  if (-f $vocab_path) {
    eval {
      my $ref = retrieve($vocab_path);
      if (ref $ref eq 'HASH') {
        my %vocabulary = %{$ref};
        $vocabulary{terms} ||= {};
        $vocabulary{_doc_count} ||= 0;
        $vocabulary{_spam_count} ||= 0;
        $vocabulary{_ham_count} ||= 0;

        # Split into text terms (TF-IDF) and rule terms (binary)
        my @vocab_keys = sort grep { !/^rule:/ } keys %{ $vocabulary{terms} };
        my @rule_keys  = sort grep { /^rule:/ } keys %{ $vocabulary{terms} };
        my %vocab_index = map { $vocab_keys[$_] => $_ } 0..$#vocab_keys;

        # Precompute IDF for text terms only
        my $N = $vocabulary{_doc_count} || 1;
        my %idf;
        foreach my $w (@vocab_keys) {
          my $df = $vocabulary{terms}{$w}{docs} || 0;
          $idf{$w} = log( ($N + 1) / ($df + 1) ) + 1;
        }

        $self->{nn_vocab} = {
          vocab_keys  => \@vocab_keys,
          vocab_index => \%vocab_index,
          rule_keys   => \@rule_keys,
          idf         => \%idf,
          spam_count  => $vocabulary{_spam_count},
          ham_count   => $vocabulary{_ham_count},
        };
      }
      1;
    } or do {
      my $err = $@ || 'unknown';
      info("Failed to load vocabulary from $vocab_path: $err");
    };
  }
}

# Tokenize text into a list of filtered tokens.
# This method is public so that sa-fann-train can use the same tokenizer.
sub tokenize_text {
    my ($self, $text, $prefix) = @_;
    my $conf = $self->{main}->{conf};

    my $min_word_len = $conf->{fann_min_word_len};
    my $max_word_len = $conf->{fann_max_word_len};
    my %stopwords    = map { lc($_) => 1 } split /\s+/, $conf->{fann_stopwords};

    return () unless defined $text;
    $text = lc $text;
    # Strip subject prefixes
    $text =~ s/^(?:[a-z]{2,12}:\s*){1,10}//i;

    # Strip anything that looks like url or email
    $text =~ s/https?(?:\:\/\/|:&#x2F;&#x2F;|%3A%2F%2F)\S{1,1024}/ /gs;
    $text =~ s/\S{1,64}?\@[a-zA-Z]\S{1,128}/ /gs;
    $text =~ s/\bwww\.\S{1,128}/ /gs;
    # Remove extra chars
    $text =~ s/\-{2,}//g;
    # Remove tokens that could be a date
    $text =~ s/\b\d+(?:\-|\/)\d+(?:\-|\/)\d+\b//g;
    # Replace HTML entities and punctuation with spaces
    $text =~ s/&[a-z#0-9]+;/ /g;
    $text =~ s{[^\p{L}\p{N}\-]}{ }g;
    my @tokens = grep { length($_) >= $min_word_len && length($_) <= $max_word_len } split /\s+/, $text;
    @tokens = grep { $_ !~ /^\d+$/ } @tokens;         # drop pure numbers
    @tokens = grep { !$stopwords{$_} } @tokens;        # drop stopwords
    if (defined $prefix && length $prefix) {
      @tokens = map { $prefix . $_ } @tokens;
    }
    return @tokens;
}

# Tokenize a filename into filtered tokens with a given prefix.
# Handles camelCase splitting and strips numbers/symbols — no URL or
# subject-prefix stripping like tokenize_text.
sub tokenize_filename {
    my ($self, $name, $prefix) = @_;
    my $conf = $self->{main}->{conf};

    my $max_word_len = $conf->{fann_max_word_len};
    my %stopwords    = map { lc($_) => 1 } split /\s+/, $conf->{fann_stopwords};

    return () unless defined $name && length $name;

    # Split camelCase: insert space before uppercase preceded by lowercase
    $name =~ s/([a-z])([A-Z])/$1 $2/g;
    # Replace numbers and non-letter chars with spaces
    $name =~ s/[^a-zA-Z]/ /g;
    $name = lc $name;

    my @tokens = grep { length($_) >= 2 && length($_) <= $max_word_len } split /\s+/, $name;
    @tokens = grep { !$stopwords{$_} } @tokens;
    if (defined $prefix && length $prefix) {
        @tokens = map { $prefix . $_ } @tokens;
    }
    return @tokens;
}

# Extract all tokens from a message for FANN classification.
# This method is public so that sa-fann-train can use the same tokenization.
sub extract_tokens {
    my ($self, $msg, $pms) = @_;
    my @tokens;

    # Body text
    my $text_arr = $msg->get_visible_rendered_body_text_array();
    my $text = join("\n", @{$text_arr});
    if (defined $text && length $text) {
        push @tokens, $self->tokenize_text($text, 'body:');
    }

    # From display name
    my $from_name = $pms->get('From:name');
    if (defined $from_name && length $from_name) {
        push @tokens, $self->tokenize_text($from_name, 'from:');
    }

    # Subject
    my $subject = $pms->get('Subject');
    if (defined $subject && length $subject) {
        push @tokens, $self->tokenize_text($subject, 'subj:');
    }

    # Attachment filenames
    foreach my $part ($msg->find_parts(qr/./, 1)) {
        my $name = $part->{name};
        next unless defined $name && length $name;
        push @tokens, $self->tokenize_filename($name, 'attach:');
    }

    # Link texts and URI TLDs
    my $uri_detail = $pms->get_uri_detail_list();
    if ($uri_detail) {
        for my $uri_info (values %$uri_detail) {
            if ($uri_info->{text}) {
                for my $txt (@{$uri_info->{text}}) {
                    next unless defined $txt && length $txt;
                    push @tokens, $self->tokenize_text($txt, 'link:');
                }
            }
            if ($uri_info->{domains}) {
                for my $domain (keys %{$uri_info->{domains}}) {
                    (my $tld = lc $domain) =~ s/^[^.]+\.//;
                    push @tokens, "utld:$tld";
                }
            }
        }
    }

    # From TLD
    my $from_addr = $pms->get('From:addr');
    if (defined $from_addr && $from_addr =~ /\@([a-zA-Z0-9._-]+)\s*$/) {
        my $raw_domain = lc($1);
        my $reg_domain = $self->{main}->{registryboundaries}->trim_domain($raw_domain);
        (my $tld = $reg_domain) =~ s/^[^.]+\.//;
        push @tokens, "ftld:$tld";
    }

    # Relay country codes
    my $relay_countries = $pms->get_tag('RELAYCOUNTRYEXT');
    if (defined $relay_countries && length $relay_countries) {
        my %seen_cc;
        for my $cc (split(/\s+/, $relay_countries)) {
            next if $cc eq '**' || !length($cc);
            $cc = uc $cc;
            push @tokens, "cc:$cc" unless $seen_cc{$cc}++;
        }
    }

    return \@tokens;
}

# Compute a TF-IDF feature vector from token list against a given vocabulary.
# This method is public so that sa-fann-train can use the same vector computation.
sub compute_tfidf_vector {
    my ($self, $tokens_ref, $vocab_keys_ref, $vocab_index_ref, $idf_ref) = @_;

    my $vocab_size = scalar @$vocab_keys_ref;
    return [] unless $vocab_size > 0;

    my %tf;
    $tf{$_}++ for @$tokens_ref;
    my $token_count = scalar @$tokens_ref || 1;

    # Build raw tf-idf vector
    my @vec = (0) x $vocab_size;
    foreach my $term (keys %tf) {
      next unless exists $vocab_index_ref->{$term};
      my $i = $vocab_index_ref->{$term};
      my $tf_val = $tf{$term} / $token_count;  # normalized TF
      $vec[$i] = $tf_val * ($idf_ref->{$term} || 1);
    }
    # L2 normalization
    my $norm = 0;
    $norm += $_ * $_ for @vec;
    $norm = sqrt($norm) || 1;
    @vec = map { $_ / $norm } @vec;

    return \@vec;
}

# check_fann(low_threshold, high_threshold)
# Registers a rule with its threshold range. The rule fires in check_cleanup
# if the prediction falls within [low_threshold, high_threshold].
# Example usage in config:
#   body FANN_SPAM_HI  eval:check_fann(0.85, 1.00)
#   body FANN_HAM_HI   eval:check_fann(0.00, 0.15)
sub check_fann {
  my ($self, $pms, $body, $low, $high) = @_;
  my $rulename = $pms->get_current_eval_rule_name();
  $pms->{fann_rules} ||= [];
  push @{$pms->{fann_rules}}, { name => $rulename, low => $low, high => $high };
  return 0;
}

sub check_cleanup {
  my ($self, $opts) = @_;
  return if $self->{training_mode};  # skip inference during sa-fann-train
  my $pms = $opts->{permsgstatus};
  $self->_run_fann_prediction($pms);

  if (defined $pms->{fann_prediction} && $pms->{fann_rules}) {
    my $p = $pms->{fann_prediction};
    for my $rule (@{$pms->{fann_rules}}) {
      if ($p >= $rule->{low} && $p <= $rule->{high}) {
        $pms->got_hit($rule->{name}, '', ruletype => 'eval');
      }
    }
    dbg("Prediction: $p");
  }
}

sub _run_fann_prediction {
  my ($self, $pms) = @_;

  return if exists $pms->{fann_prediction};
  return if $self->{training_mode};  # skip inference during sa-fann-train

  my $msg = $pms->{msg};
  my $conf = $self->{main}->{conf};

  # Use cached vocabulary and model from finish_parsing_end
  my $vocab = $self->{nn_vocab};
  my $network = $self->{neural_model};
  unless ($vocab && $network) {
    $pms->{fann_prediction} = undef;
    dbg("No vocabulary or model loaded");
    return;
  }

  # Ensure we have enough spam and ham examples
  my $min_spam = $conf->{fann_min_spam_count};
  my $min_ham  = $conf->{fann_min_ham_count};
  if ( ($vocab->{spam_count} < $min_spam) || ($vocab->{ham_count} < $min_ham) ) {
    dbg("Insufficient spam/ham data for prediction: spam=".$vocab->{spam_count}.", ham=".$vocab->{ham_count});
    $pms->{fann_prediction} = undef;
    return;
  }

  my $vocab_keys  = $vocab->{vocab_keys};
  my $vocab_index = $vocab->{vocab_index};
  my $idf         = $vocab->{idf};
  my $vocab_size  = scalar @$vocab_keys;

  # Extract tokens and compute TF-IDF vector
  my @tfidf_vec;
  if ($vocab_size > 0) {
    my @tokens = @{ $self->extract_tokens($msg, $pms) };
    my $vec = $self->compute_tfidf_vector(\@tokens, $vocab_keys, $vocab_index, $idf);
    @tfidf_vec = @$vec;
  } else {
    @tfidf_vec = ();
  }

  # Append binary rule features
  my $rule_keys = $vocab->{rule_keys};
  my @rule_vec;
  if (@$rule_keys) {
    my %hits;
    my $hit_str = $pms->get_names_of_tests_hit();
    if ($hit_str) {
      $hits{$_} = 1 for split(/,/, $hit_str);
    }
    my $sub_str = $pms->get_names_of_subtests_hit();
    if ($sub_str) {
      $hits{$_} = 1 for split(/,/, $sub_str);
    }
    @rule_vec = map { $hits{ substr($_, 5) } ? 1 : 0 } @$rule_keys;
  }

  my @combined = (@tfidf_vec, @rule_vec);
  unless (@combined) {
    $pms->{fann_prediction} = undef;
    dbg("No features available");
    return;
  }

  my $expected_size = $network->num_inputs();
  if (scalar(@combined) != $expected_size) {
    dbg("Vocabulary size mismatch (got ".scalar(@combined).", model expects ".$expected_size."), run sa-fann-train to retrain");
    $pms->{fann_prediction} = undef;
    return;
  }

  my $prediction = eval { $network->run(\@combined) } ;
  if ($@) {
    $pms->{fann_prediction} = undef;
    dbg("Prediction failed: $@");
    return;
  }
  $prediction = ref($prediction) ? $prediction->[0] : $prediction;

  unless(defined $prediction) {
    dbg("No prediction available");
    $pms->{fann_prediction} = undef;
    return;
  }

  dbg("Prediction: $prediction");
  $pms->{fann_prediction} = $prediction;
}

1;

=head1 SEE ALSO

L<sa-fann-train(1)>

=head1 LICENSE

Copyright 2026 MXGuardian

Licensed under the Apache License, Version 2.0.

=cut
