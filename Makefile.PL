use strict;
use warnings;
use ExtUtils::MakeMaker;

WriteMakefile(
    NAME               => 'Mail::SpamAssassin::Plugin::FANN',
    VERSION_FROM       => 'lib/Mail/SpamAssassin/Plugin/FANN.pm',
    ABSTRACT_FROM      => 'lib/Mail/SpamAssassin/Plugin/FANN.pm',
    AUTHOR             => 'MXGuardian',
    LICENSE            => 'apache_2_0',
    MIN_PERL_VERSION   => '5.014',
    EXE_FILES          => ['bin/sa-fann-train'],
    PREREQ_PM          => {
        'Mail::SpamAssassin' => '4.0',
        'AI::FANN'           => 0,
        'Storable'           => 0,
    },
    TEST_REQUIRES      => {
        'Test::More' => 0,
    },
    META_MERGE         => {
        'meta-spec' => { version => 2 },
        resources   => {
            repository => {
                type => 'git',
                url  => 'https://github.com/mxguardian/Mail-SpamAssassin-Plugin-FANN.git',
                web  => 'https://github.com/mxguardian/Mail-SpamAssassin-Plugin-FANN',
            },
        },
    },
);
