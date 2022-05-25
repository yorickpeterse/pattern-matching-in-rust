# ML Pattern match compilation and partial evaluation

This directory contains an implementation of the pattern matching algorithm
introduced in the paper "ML Pattern match compilation and partial evaluation" by
Peter Sestoft, from 1996.

## A short rant about the paper

The paper is a bit of a pain to read, and took me a solid week to understand.
Part of this is because I'm not familiar with Standard ML, so I first had to
learn that to some degree. The syntax can also be hard to grok when you have
functions calling functions and passing those results as arguments directly,
especially combined with operators (e.g is `foo bar :: baz` parsed as
`foo(bar :: baz)` or `(foo bar) :: baz`?).

It doesn't help that the paper makes references to the author's implementation,
but the only two links regarding it are dead FTP links. I eventually found these
implementations of this algorithm:

- https://github.com/kfl/mosml/blob/f529b33bb891ff1df4aab198edad376f9ff64d28/src/compiler/Match.sml
- https://github.com/rsdn/nemerle/blob/db4bc9078f1b6238da32df1519c1957e74b6834a/ncc/typing/DecisionTreeBuilder.n
- https://github.com/rsdn/nitra/tree/master/Nitra/Nitra.Compiler/Generation/PatternMatching
- https://github.com/melsman/mlkit/blob/237be62778985e76f912cefdc0bb21b22bed5bd4/src/Compiler/Lambda/CompileDec.sml#L510

The Moscow ML implementation uses memoization and some extensions for the
pattern matching logic. The Nemerle implementation is quite different and uses a
more imperative/mutable approach.

As to how the algorithm works: even now I don't quite understand why certain
decisions were made, and the algorithm as a whole feels a bit crude.

I could go on, but the summary is this: if you wish to understand the paper, I
recommend reading through it while using my Rust code as a reference. It should
be a bit easier to understand and translate to other languages, and it doesn't
require a 20 year old language (though maybe it will if you're reading this 20
years from now).

## Project structure

There are two implementations of the algorithm: a raw version, and an idiomatic
version. Neither version implements the memoization strategy as discussed in
section 7.5, as this likely won't work well due to Rust's single ownership
requirement. Both versions are extensively commented to better explain why
certain decisions where made, what to keep in mind when reading the paper, etc.

### The raw version

The raw version is more or less 1:1 translation of the SML code included in the
paper. The code is terrible, relies on (poorly implemented) immutable lists
(because the original algorithm requires immutable lists), and likely performs
extremely poorly. I tried to keep this version as close to the paper as
possible, only deviating where Rust simply required a different approach.

Some differences from the paper:

- Rust doesn't have built-in immutable lists, and the algorithm requires the use
  of immutable lists in a few places. Thus, we introduce a custom immutable
  linked list.
- The paper assumes multiple ownership of values in a few places. This
  implementation instead clones values to work around that, as using a different
  approach requires a different implementation.
- The `succeed'` and `fail'` functions are called `match_succeed` and
  `match_fail` respectively. Who the hell thought it was a good idea to allow
  quotes in symbol names?
- When generating `Sel` nodes, the paper uses `i+1` to build the selector
  values. It's not clear why this is done (the paper makes no mention of it),
  and it seems unnecessary. As such we just use indexes starting at zero.
- The paper implements various functions in an non-exhaustive manner, without
  any explanation as to why. My implementation uses exhaustive patterns where
  possible, and `unwrap()` in a few places where missing values (and thus
  panics) shouldn't occur in the absence of bugs (famous last words).

### The idiomatic version

This implementation of the algorithm is closer to what you'd normally write in
Rust. Some of the names used are still a bit confusing, but unfortunately I
haven't been able to come up with better names.

Unlike the raw implementation, this implementation doesn't rely on persistent
lists. Instead, it uses mutable vectors that store values in reverse order.
Storing them in this order means a pop() returns the head of the vector, instead
of the tail. This makes retrieving the head cheap, as no values need to be
shifted.

Some function (e.g. `addneg` and `match_fail`) are inlined into their callers,
as they are only called in one place.

For traversing all the pattern matching rules we use a cursor, essentially
turning the list into an iterator that you can rewind. This is needed because
when building an `IfEq` node, both the true and false bodies need to start off
with the same set of rules. Using a cursor allows us to do just that, but
without cloning the rules.
