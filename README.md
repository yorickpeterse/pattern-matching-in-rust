# Pattern matching in Rust

This repository contains a collection of pattern matching algorithms implemented
in Rust. The goal of these implementations it to (hopefully) make it easier to
understand them, as papers related to pattern matching (and papers in general)
can be difficult to read.

## Algorithms

| Name                                          | Paper                        | Directory
|:----------------------------------------------|:-----------------------------|:-----------
| ML pattern compilation and partial evaluation | [PDF](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.48.1363) | [sestoft1996][sestoft1996/] |

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
