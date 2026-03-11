#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir);
use File::Spec;
use File::Copy;
use FindBin;

my $libdir    = File::Spec->catdir($FindBin::Bin, '..', 'lib');
my $staticcf  = File::Spec->catdir($FindBin::Bin, 'config');

# Ensure we can load the plugin and SpamAssassin
eval { require Mail::SpamAssassin; 1 }
  or plan skip_all => 'Mail::SpamAssassin not installed';
eval { require AI::FANN; 1 }
  or plan skip_all => 'AI::FANN not installed';

plan tests => 3;

my $datadir  = File::Spec->catdir($FindBin::Bin, 'data');
my $spamdir  = File::Spec->catdir($datadir, 'spam');
my $hamdir   = File::Spec->catdir($datadir, 'ham');
my $tmpdir   = tempdir(CLEANUP => 1);
my $fanndir  = File::Spec->catdir($tmpdir, 'fann');
mkdir $fanndir;

# Build config dir in tmpdir: copy static configs + add fann_data_dir
my $cfdir = File::Spec->catdir($tmpdir, 'config');
mkdir $cfdir;
for my $f (qw(init.pre local.cf)) {
  copy(File::Spec->catfile($staticcf, $f), File::Spec->catfile($cfdir, $f))
    or die "Cannot copy $f: $!";
}
# Append fann_data_dir to local.cf
open my $fh, '>>', File::Spec->catfile($cfdir, 'local.cf') or die $!;
print $fh "fann_data_dir\t$fanndir\n";
close $fh;

# Empty user_prefs to prevent loading from ~/.spamassassin
my $prefs = File::Spec->catfile($tmpdir, 'user_prefs');
open $fh, '>', $prefs or die $!;
close $fh;

# Train model
my $train_script = File::Spec->catfile($FindBin::Bin, '..', 'bin', 'sa-fann-train');
my $cmd = "$^X -T -I$libdir" .
          " $train_script -L" .
          " --spam $spamdir --ham $hamdir" .
          " -C $cfdir --siteconfigpath $cfdir" .
          " -p $prefs";
diag("Training: $cmd");
my $rc = system($cmd);
ok($rc == 0, 'sa-fann-train completes successfully');

# Helper: scan a message and return the hit string
sub scan_message {
  my ($file) = @_;

  unshift @INC, $libdir unless grep { $_ eq $libdir } @INC;

  my $sa = Mail::SpamAssassin->new({
    rules_filename      => $cfdir,
    site_rules_filename => $cfdir,
    userprefs_filename  => $prefs,
    local_tests_only    => 1,
    dont_copy_prefs     => 1,
  });
  $sa->init(1);

  open my $mfh, '<', $file or die "Cannot open $file: $!";
  my $raw = do { local $/; <$mfh> };
  close $mfh;

  my $msg = $sa->parse(\$raw);
  my $pms = $sa->check($msg);
  my $hits = $pms->get_names_of_tests_hit() || '';
  $pms->finish();
  $msg->finish();
  $sa->finish();
  return $hits;
}

# Pick first spam and ham file (skip dotfiles)
opendir my $dh, $spamdir or die "Cannot open $spamdir: $!";
my ($spam_file) = map { File::Spec->catfile($spamdir, $_) }
                  sort grep { !/^\./ && -f File::Spec->catfile($spamdir, $_) } readdir($dh);
closedir $dh;

opendir $dh, $hamdir or die "Cannot open $hamdir: $!";
my ($ham_file) = map { File::Spec->catfile($hamdir, $_) }
                 sort grep { !/^\./ && -f File::Spec->catfile($hamdir, $_) } readdir($dh);
closedir $dh;

# Verify spam is detected as spam
my $spam_hits = scan_message($spam_file);
diag("Spam hits: $spam_hits");
ok($spam_hits =~ /\bFANN_SPAM\b/ && $spam_hits !~ /\bFANN_HAM\b/,
   "spam detected as spam");

# Verify ham is detected as ham
my $ham_hits = scan_message($ham_file);
diag("Ham hits: $ham_hits");
ok($ham_hits =~ /\bFANN_HAM\b/ && $ham_hits !~ /\bFANN_SPAM\b/,
   "ham detected as ham");
