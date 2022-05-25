/// An idiomatic Rust implementation of the pattern matching algorithm.
use std::collections::HashSet;

/// The result of a static match.
#[derive(Debug)]
enum Match {
    Yes,
    No,
    Maybe,
}

/// The description of terms already matched, corresponding to the `context`
/// type in the paper.
struct Context {
    values: Vec<(Constructor, Vec<Term>)>,
}

impl Context {
    fn new() -> Self {
        Self { values: Vec::new() }
    }

    fn push(&mut self, value: (Constructor, Vec<Term>)) {
        self.values.push(value);
    }

    fn pop(&mut self) -> Option<(Constructor, Vec<Term>)> {
        self.values.pop()
    }

    fn add_argument_to_last(&mut self, term: Term) {
        if let Some((_, args)) = self.values.last_mut() {
            args.push(term);
        }
    }

    fn reconstruct_term(&self, term: Term, work: &Work) -> Term {
        self.values.iter().zip(work.iter()).fold(
            term,
            |term, ((con, args), (_, _, dargs))| {
                let mut new_args: Vec<_> = dargs.clone();

                new_args.push(term);
                new_args.extend(args.iter().rev().cloned());
                Term::Pos(con.clone(), new_args)
            },
        )
    }
}

/// The work stack as used in the paper.
///
/// The paper uses a list of triple lists, removing the need for some append
/// operations. This is a bit annoying to work with in Rust (we have to unwrap()
/// in some places), but again we're trying to stay as close to the paper as
/// possible.
///
/// We use a type alias here so we don't have to re-type this type name in the
/// various places that it's used.
type Work = Vec<(Vec<Pattern>, Vec<Access>, Vec<Term>)>;

/// The type of the right-hand side of a case (i.e. the code to run).
///
/// For the sake of simplicity we just use a String here. In a real compiler
/// this would probably be an AST node or another sort of IR to run upon a
/// match.
pub type RHS = String;

/// A type for storing diagnostics produced by the decision tree compiler.
pub struct Diagnostics {
    /// The diagnostic messages produced.
    ///
    /// In a real compiler this would include more than just a message, such as
    /// the line and numbers.
    messages: Vec<String>,

    /// The right-hand values (= the code you'd run upon a match) that have been
    /// processed.
    ///
    /// If a value isn't included in this set it means it and its pattern are
    /// redundant.
    ///
    /// In a real compiler you'd probably mark AST nodes directly. In our case
    /// the right-hand values are just simple strings, so we use a set instead.
    reachable: HashSet<RHS>,
}

/// A type for compiling a list of rules into a decision tree.
pub struct Compiler {
    /// The rules to compile into a decision tree.
    rules: Vec<(Pattern, RHS)>,

    /// The start of the first rule to compile.
    ///
    /// When generating IfEq nodes we need to generate two branches, both
    /// starting with the same set of rules. To avoid cloning we use a cursor,
    /// save it before processing one branch, then restore it for the other
    /// branch.
    rules_index: usize,

    diagnostics: Diagnostics,
}

impl Compiler {
    pub fn new(rules: Vec<(Pattern, RHS)>) -> Self {
        Self {
            rules,
            rules_index: 0,
            diagnostics: Diagnostics {
                messages: Vec::new(),
                reachable: HashSet::new(),
            },
        }
    }

    pub fn compile(&mut self) -> Decision {
        self.fail(Term::bottom())
    }

    fn fail(&mut self, term: Term) -> Decision {
        if let Some((pat, rhs)) = self.next_rule().cloned() {
            let ctx = Context::new();
            let work = Vec::new();

            self.matches(pat, Access::Obj, term, ctx, work, rhs)
        } else {
            self.diagnostics
                .messages
                .push(format!("Missing pattern: {}", term.error_string()));

            Decision::Failure
        }
    }

    fn succeed(
        &mut self,
        mut ctx: Context,
        mut work: Work,
        rhs: RHS,
    ) -> Decision {
        if let Some((mut pats, mut accs, mut terms)) = work.pop() {
            if let (Some(pat), Some(acc), Some(term)) =
                (pats.pop(), accs.pop(), terms.pop())
            {
                work.push((pats, accs, terms));
                self.matches(pat, acc, term, ctx, work, rhs)
            } else {
                if let Some((con, mut args)) = ctx.pop() {
                    args.reverse();
                    ctx.add_argument_to_last(Term::Pos(con, args));
                }

                self.succeed(ctx, work, rhs)
            }
        } else {
            self.diagnostics.reachable.insert(rhs.clone());
            Decision::Success(rhs)
        }
    }

    fn matches(
        &mut self,
        pattern: Pattern,
        access: Access,
        term: Term,
        mut ctx: Context,
        work: Work,
        rhs: RHS,
    ) -> Decision {
        match pattern {
            Pattern::Variable(_) => {
                ctx.add_argument_to_last(term);
                self.succeed(ctx, work, rhs)
            }
            Pattern::Constructor(con, args) => match self
                .match_term(&con, &term)
            {
                Match::Yes => {
                    self.build_match(con, args, access, term, ctx, work, rhs)
                }
                Match::No => self.fail(ctx.reconstruct_term(term, &work)),
                Match::Maybe => {
                    let false_term =
                        ctx.reconstruct_term(term.clone().negated(&con), &work);
                    let cursor = self.rules_index;
                    let matched = self.build_match(
                        con.clone(),
                        args,
                        access.clone(),
                        term,
                        ctx,
                        work,
                        rhs,
                    );

                    self.rules_index = cursor;

                    Decision::IfEq(
                        access,
                        con,
                        Box::new(matched),
                        Box::new(self.fail(false_term)),
                    )
                }
            },
        }
    }

    fn build_match(
        &mut self,
        con: Constructor,
        args: Vec<Pattern>,
        obj: Access,
        term: Term,
        mut ctx: Context,
        mut work: Work,
        rhs: RHS,
    ) -> Decision {
        let access = (0..con.arity)
            .rev()
            .map(|i| Access::Sel(i, Box::new(obj.clone())))
            .collect();

        let terms = match term {
            Term::Pos(_, dargs) => dargs,
            Term::Neg(_) => vec![Term::bottom(); con.arity],
        };

        ctx.push((con, Vec::new()));
        work.push((args, access, terms));
        self.succeed(ctx, work, rhs)
    }

    fn match_term(&mut self, con: &Constructor, term: &Term) -> Match {
        match term {
            Term::Pos(scon, _) if con == scon => Match::Yes,
            Term::Pos(_, _) => Match::No,
            Term::Neg(exl) if exl.contains(con) => Match::No,
            Term::Neg(exl) if con.span == (exl.len() + 1) => Match::Yes,
            Term::Neg(_) => Match::Maybe,
        }
    }

    fn next_rule(&mut self) -> Option<&(Pattern, RHS)> {
        if self.rules_index >= self.rules.len() {
            None
        } else {
            let val = self.rules.get(self.rules_index);

            self.rules_index += 1;

            val
        }
    }
}

/// A type constructor.
///
/// For a boolean, a constructor would have the following values:
///
/// - name: true or false
/// - arity: 0, as booleans don't take arguments
/// - span: 2, as there are only two constructors (true and false)
///
/// In a real compiler you'd probably use pointers/IDs to your type data
/// structures instead, but for the sake of keeping things simple we just use a
/// struct that can be cloned.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Constructor {
    name: String,

    // The number of arguments.
    arity: usize,

    // The total number of constructors of the owning type
    //
    // A span of 0 means the type has an infinite amount of constructors.
    span: usize,
}

/// A user provided pattern to match against an input value.
///
/// We only provide two types of patterns: constructors, and variables/bindings.
///
/// In a real compiler you'd probably be using AST nodes instead of dedicated
/// pattern types, and include more cases for specific patterns (e.g. tuple and
/// struct patterns).
#[derive(Debug, Clone)]
pub enum Pattern {
    Constructor(Constructor, Vec<Pattern>),
    Variable(String),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Term {
    // `Cons` is the top-most constructor, and its components are described by
    // the Vec.
    //
    // The arguments are in reverse order, so the first argument is the last
    // value.
    Pos(Constructor, Vec<Term>),

    // Any term who's top-most constructor is _not_ any of the listed
    // constructors.
    //
    // For a Negative(S), the cardinality of S must be less than the span of
    // any constructor in S:
    //
    //     cons.iter().all(|cons| cardinality(s) < span(cons))
    //
    // Due to static typing, all constructors in S are of the same type, thus
    // have the same span.
    //
    // The constructors are in reverse order, so the first constructor is the
    // last value.
    Neg(Vec<Constructor>),
}

impl Term {
    fn bottom() -> Term {
        Term::Neg(Vec::new())
    }

    fn negated(self, con: &Constructor) -> Term {
        match self {
            Term::Pos(_, _) => self,
            Term::Neg(mut nonset) => {
                nonset.push(con.clone());
                Term::Neg(nonset)
            }
        }
    }
}

impl Term {
    /// Returns a string used to describe this term in an error message.
    fn error_string(&self) -> String {
        match self {
            Term::Pos(cons, args) => {
                if args.is_empty() {
                    cons.name.clone()
                } else {
                    format!(
                        "{}({})",
                        cons.name,
                        args.iter()
                            .rev()
                            .map(|v| v.error_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
            Term::Neg(_) => "_".to_string(),
        }
    }
}

/// The `access` type in the paper.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Access {
    Obj,
    Sel(usize, Box<Access>),
}

/// The `decision` type in the paper.
#[derive(Debug, Eq, PartialEq, Clone)]
pub enum Decision {
    /// A pattern didn't match.
    Failure,

    /// A pattern is matched and the right-hand value is to be returned.
    Success(RHS),

    /// Checks if a constructor matches the value at the given access path.
    IfEq(Access, Constructor, Box<Decision>, Box<Decision>),

    /// Checks if any of the given constructors match the value at the given
    /// access path.
    Switch(Access, Vec<(Constructor, Decision)>, Box<Decision>),
}

impl Decision {
    /// Replaces a series of nested IfEq nodes for the same access object with a
    /// Switch node.
    pub fn replace_nested_if(self) -> Decision {
        match self {
            Decision::IfEq(root, con, ok, fail) => {
                let mut cases = vec![(con, *ok)];
                let mut fallback = fail;

                loop {
                    match *fallback {
                        Decision::IfEq(acc, con, ok, fail) if root == acc => {
                            fallback = fail;

                            cases.push((con, *ok));
                        }
                        _ => break,
                    }
                }

                if cases.len() == 1 {
                    let (con, ok) = cases.pop().unwrap();

                    Decision::IfEq(root, con, Box::new(ok), fallback)
                } else {
                    Decision::Switch(root, cases, fallback)
                }
            }
            _ => self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn con(name: &str, arity: usize, span: usize) -> Constructor {
        Constructor { name: name.to_string(), arity, span }
    }

    fn nil() -> Pattern {
        Pattern::Constructor(con("nil", 0, 1), Vec::new())
    }

    fn tt_con() -> Constructor {
        con("true", 0, 2)
    }

    fn ff_con() -> Constructor {
        con("false", 0, 2)
    }

    fn tt() -> Pattern {
        Pattern::Constructor(tt_con(), Vec::new())
    }

    fn ff() -> Pattern {
        Pattern::Constructor(ff_con(), Vec::new())
    }

    fn pair(a: Pattern, b: Pattern) -> Pattern {
        Pattern::Constructor(con("pair", 2, 1), vec![b, a])
    }

    fn var(name: &str) -> Pattern {
        Pattern::Variable(name.to_string())
    }

    fn if_eq(
        acc: Access,
        con: Constructor,
        ok: Decision,
        fail: Decision,
    ) -> Decision {
        Decision::IfEq(acc, con, Box::new(ok), Box::new(fail))
    }

    fn switch(
        acc: Access,
        cases: Vec<(Constructor, Decision)>,
        fallback: Decision,
    ) -> Decision {
        Decision::Switch(acc, cases, Box::new(fallback))
    }

    fn success(value: &str) -> Decision {
        Decision::Success(value.to_string())
    }

    fn failure() -> Decision {
        Decision::Failure
    }

    fn rhs(value: &str) -> String {
        value.to_string()
    }

    fn obj() -> Access {
        Access::Obj
    }

    fn sel(index: usize, acc: Access) -> Access {
        Access::Sel(index, Box::new(acc))
    }

    fn compile(rules: Vec<(Pattern, RHS)>) -> (Decision, Diagnostics) {
        let mut compiler = Compiler::new(rules);
        let tree = compiler.compile();

        (tree, compiler.diagnostics)
    }

    #[test]
    fn test_term_description_error_string() {
        let term = Term::Pos(
            con("box", 2, 1),
            vec![
                Term::Neg(vec![con("false", 0, 2)]),
                Term::Pos(con("true", 0, 2), Vec::new()),
            ],
        );

        assert_eq!(term.error_string(), "box(true, _)");
    }

    #[test]
    fn test_context_reconstruct_term() {
        let mut ctx = Context::new();

        ctx.push((
            con("baz", 0, 1),
            vec![
                Term::Neg(vec![con("arg2", 0, 1)]),
                Term::Neg(vec![con("arg1", 0, 1)]),
            ],
        ));

        let work = vec![(
            Vec::new(),
            Vec::new(),
            vec![
                Term::Neg(vec![con("work2", 0, 1)]),
                Term::Neg(vec![con("work1", 0, 1)]),
            ],
        )];
        let dsc = Term::Neg(vec![con("bar", 0, 1)]);
        let new_dsc = ctx.reconstruct_term(dsc, &work);

        assert_eq!(
            new_dsc,
            Term::Pos(
                con("baz", 0, 1),
                vec![
                    Term::Neg(vec![con("work2", 0, 1)]),
                    Term::Neg(vec![con("work1", 0, 1)]),
                    Term::Neg(vec![con("bar", 0, 1)]),
                    Term::Neg(vec![con("arg1", 0, 1)]),
                    Term::Neg(vec![con("arg2", 0, 1)]),
                ]
            )
        );
    }

    #[test]
    fn test_context_add_argument_to_last() {
        let mut ctx = Context::new();

        ctx.push((
            con("baz", 0, 1),
            vec![
                Term::Neg(vec![con("arg2", 0, 1)]),
                Term::Neg(vec![con("arg1", 0, 1)]),
            ],
        ));

        let term = Term::Neg(vec![con("bar", 0, 1)]);

        ctx.add_argument_to_last(term);

        assert_eq!(
            ctx.values,
            vec![(
                con("baz", 0, 1),
                vec![
                    Term::Neg(vec![con("arg2", 0, 1)]),
                    Term::Neg(vec![con("arg1", 0, 1)]),
                    Term::Neg(vec![con("bar", 0, 1)]),
                ]
            )]
        );
    }

    #[test]
    fn test_match_always_succeeds() {
        let (result, _) = compile(vec![(nil(), rhs("true"))]);

        assert_eq!(result, success("true"));
    }

    #[test]
    fn test_match_always_fails() {
        let (result, _) = compile(Vec::new());

        assert_eq!(result, failure());
    }

    #[test]
    fn test_match_single_pattern() {
        let (result, _) =
            compile(vec![(tt(), rhs("true")), (ff(), rhs("false"))]);

        assert_eq!(
            result,
            if_eq(obj(), tt_con(), success("true"), success("false"))
        );
    }

    #[test]
    fn test_match_var() {
        let (result, _) = compile(vec![(var("a"), rhs("true"))]);

        assert_eq!(result, success("true"));
    }

    #[test]
    fn test_match_multiple_patterns() {
        let (result, diags) = compile(vec![
            (tt(), rhs("true")),
            (ff(), rhs("false")),
            (tt(), rhs("redundant")),
        ]);

        // Redundant patterns are ignored on the decision tree. This is also how
        // you'd detect redundant patterns: you'd somehow mark every RHS when
        // you produce their Success nodes. Any RHS nodes that remain unmarked
        // are redundant.
        assert_eq!(
            result,
            if_eq(obj(), tt_con(), success("true"), success("false"))
        );

        assert!(diags.reachable.contains(&"true".to_string()));
        assert!(diags.reachable.contains(&"false".to_string()));
        assert!(!diags.reachable.contains(&"redundant".to_string()));
    }

    #[test]
    fn test_nonexhaustive_match() {
        let (result, diags) = compile(vec![(tt(), rhs("true"))]);

        assert_eq!(result, if_eq(obj(), tt_con(), success("true"), failure()));
        assert_eq!(diags.messages, vec!["Missing pattern: _".to_string()]);
    }

    #[test]
    fn test_nonexhaustive_match_from_paper() {
        let green = Pattern::Constructor(con("green", 0, 3), Vec::new());
        let (result, diags) = compile(vec![
            (pair(tt(), green.clone()), rhs("111")),
            (pair(ff(), green.clone()), rhs("222")),
        ]);

        assert_eq!(
            result,
            if_eq(
                sel(0, obj()),
                tt_con(),
                if_eq(
                    sel(1, obj()),
                    con("green", 0, 3),
                    success("111"),
                    failure()
                ),
                if_eq(
                    sel(1, obj()),
                    con("green", 0, 3),
                    success("222"),
                    failure()
                )
            )
        );

        assert_eq!(
            diags.messages,
            vec![
                "Missing pattern: pair(true, _)".to_string(),
                "Missing pattern: pair(false, _)".to_string()
            ]
        );
    }

    #[test]
    fn test_nested_match() {
        let (result, _) = compile(vec![
            (pair(tt(), tt()), rhs("foo")),
            (pair(tt(), ff()), rhs("bar")),
            (pair(ff(), ff()), rhs("baz")),
            (pair(ff(), tt()), rhs("quix")),
        ]);

        assert_eq!(
            result,
            if_eq(
                sel(0, obj()),
                tt_con(),
                if_eq(sel(1, obj()), tt_con(), success("foo"), success("bar")),
                if_eq(sel(1, obj()), ff_con(), success("baz"), success("quix"))
            )
        );
    }

    #[test]
    fn test_match_with_replace_nested_if() {
        let a = con("a", 0, 4);
        let b = con("b", 0, 4);
        let c = con("c", 0, 4);
        let d = con("d", 0, 4);
        let a_pat = Pattern::Constructor(a.clone(), Vec::new());
        let b_pat = Pattern::Constructor(b.clone(), Vec::new());
        let c_pat = Pattern::Constructor(c.clone(), Vec::new());
        let d_pat = Pattern::Constructor(d.clone(), Vec::new());
        let (result, _) = compile(vec![
            ((a_pat, rhs("a"))),
            ((b_pat, rhs("b"))),
            ((c_pat, rhs("c"))),
            ((d_pat, rhs("d"))),
        ]);

        assert_eq!(
            result.replace_nested_if(),
            switch(
                obj(),
                vec![(a, success("a")), (b, success("b")), (c, success("c"))],
                success("d")
            )
        );
    }

    #[test]
    fn test_nested_match_without_switch() {
        let (result, _) = compile(vec![
            (pair(tt(), tt()), rhs("foo")),
            (pair(tt(), ff()), rhs("bar")),
            (pair(ff(), ff()), rhs("baz")),
            (pair(ff(), tt()), rhs("quix")),
        ]);

        // This doesn't produce a switch, as the nested patterns don't test the
        // same value.
        assert_eq!(
            result.replace_nested_if(),
            if_eq(
                sel(0, obj()),
                tt_con(),
                if_eq(sel(1, obj()), tt_con(), success("foo"), success("bar")),
                if_eq(sel(1, obj()), ff_con(), success("baz"), success("quix"))
            )
        );
    }

    #[test]
    fn test_match_with_args() {
        let some = con("some", 3, 2);
        let (result, _) = compile(vec![
            (
                Pattern::Constructor(some.clone(), vec![ff(), tt(), tt()]),
                rhs("foo"),
            ),
            (var("x"), rhs("bar")),
        ]);

        assert_eq!(
            result,
            if_eq(
                obj(),
                some,
                if_eq(
                    sel(0, obj()),
                    tt_con(),
                    if_eq(
                        sel(1, obj()),
                        tt_con(),
                        if_eq(
                            sel(2, obj()),
                            ff_con(),
                            success("foo"),
                            success("bar")
                        ),
                        success("bar")
                    ),
                    success("bar")
                ),
                success("bar")
            )
        );
    }

    #[test]
    fn test_match_nonexhaustive_with_args() {
        let some = con("some", 3, 2);
        let (result, diags) = compile(vec![(
            Pattern::Constructor(some.clone(), vec![ff(), ff(), tt()]),
            rhs("foo"),
        )]);

        assert_eq!(
            result,
            if_eq(
                obj(),
                some,
                if_eq(
                    sel(0, obj()),
                    tt_con(),
                    if_eq(
                        sel(1, obj()),
                        ff_con(),
                        if_eq(
                            sel(2, obj()),
                            ff_con(),
                            success("foo"),
                            failure()
                        ),
                        failure()
                    ),
                    failure()
                ),
                failure()
            )
        );

        assert_eq!(
            diags.messages,
            vec![
                "Missing pattern: some(true, false, _)".to_string(),
                "Missing pattern: some(true, _, _)".to_string(),
                "Missing pattern: some(_, _, _)".to_string(),
                "Missing pattern: _".to_string(),
            ]
        );
    }
}
