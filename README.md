# Pattern matching in Rust

This repository contains a collection of pattern matching algorithms implemented
in Rust. The goal of these implementations it to (hopefully) make it easier to
understand them, as papers related to pattern matching (and papers in general)
can be difficult to read.

## Background

I ended up implementing these algorithms while investigating potential pattern
matching/exhaustiveness checking algorithms for [Inko](https://inko-lang.org/).
While there are plenty of papers on the subject, few of them include reference
code, and almost all of them are really dense and difficult to read. I hope the
code published in this repository is of use to those wishing to implement
pattern matching/exhaustiveness.

## Algorithms

| Name                                          | Paper                        | Directory
|:----------------------------------------------|:-----------------------------|:-----------
| ML pattern compilation and partial evaluation | [PDF](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.48.1363) | [sestoft1996](./sestoft1996/) |

Other papers I've come across (but don't necessarily want to implement):

- [A generic algorithm for checking exhaustivity of pattern
  matching](https://dl.acm.org/doi/10.1145/2998392.2998401).
  - The Scala implementation [is found in this PR](https://github.com/lampepfl/dotty/pull/1364) (the `Space.scala` file).
  - Swift also uses this algorithm [here](https://github.com/apple/swift/blob/3c0b1ab03f189e044303436b8aa6a27c2f93707d/lib/Sema/TypeCheckSwitchStmt.cpp)
  - Some Reddit comments about the algorithm are [found here](https://www.reddit.com/r/ProgrammingLanguages/comments/cioxwn/a_generic_algorithm_for_checking_exhaustivity_of/)
- [Compiling pattern matching to good decision
  trees](https://www.cs.tufts.edu/comp/150FP/archive/luc-maranget/jun08.pdf).
  This is about just compiling pattern matching into a decision tree, not about
  exhaustiveness checking. If you don't know how to read the computer science
  hieroglyphs (like me), this paper is basically impossible to understand.
- [Warnings for pattern
  matching](http://pauillac.inria.fr/~maranget/papers/warn/warn.pdf). This is
  just about producing warnings/errors for e.g. non-exhaustive patterns.
  Similarly painful to understand as the previous paper (i.e. I gave up).
- [The Implementation of Functional Programming
  Languages](https://www.microsoft.com/en-us/research/publication/the-implementation-of-functional-programming-languages/).
  This book has a chapter on pattern matching, but I gave up on it.
- [How to compile pattern
  matching](https://julesjacobs.com/notes/patternmatching/patternmatching.pdf)
  provides an algorithm somewhat different from the existing literature.
  Unfortunately, the provided Scala implementation doesn't do proper
  exhaustiveness checking. I contacted the author and while they did provide a
  version that did some degree of exhaustiveness checking, I never managed to
  properly understand it or the algorithm as a whole.

## Requirements

A recent-ish (as of 2022) Rust version that supports the 2021 edition (though I
think the 2018 edition should also work).

## Usage

Each algorithm is implemented as a library, and come with a set of unit tests
that you can run using `cargo test`.

## Licence

The code in this repository is licensed under the
[Unlicense](https://unlicense.org/). A copy of this license can be found in the
file "LICENSE".
