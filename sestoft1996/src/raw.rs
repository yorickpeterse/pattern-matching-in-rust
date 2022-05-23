// This implementation is a more or less 1:1 port of the SML implementation
// provided in the paper, with only a few changes made to make things work on
// Rust. For example, the SML implementation assumes multiple ownership of
// certain values, which isn't allowed in Rust. For the sake of simplicity, we
// just clone values in this case.
//
// This implementation doesn't use the memoization approach briefly mentioned in
// section 7.5 of the paper. This requires multiple ownership of the tree nodes,
// or a different way of building the tree/graph (e.g. using IDs). To keep
// things simple, we skip over this.
//
// Because this implementation is more or less a direct translation, it's _not_
// idiomatic Rust. An idiomatic implementation is provided separately.
//
// The Moscow ML compiler uses hash consing and a DAG as discussed in section
// 7.5 of the paper.
use std::fmt;
use std::rc::Rc;

/// An immutable linked list.
///
/// The algorithm as presented in the paper makes use of and requires immutable
/// lists. For example, when compiling the `IfEq` nodes it compiles two
/// different branches, but assumes work start off with the same set of rules,
/// `work` values, etc. Since we're trying to stay as close to the paper as
/// possible, we also follow the use of immutable data types.
///
/// Like the rest of this implementation we're focusing on keeping things as
/// simple as is reasonable, rather than making the implementation efficient.
struct Node<T> {
    value: T,
    next: Option<Rc<Node<T>>>,
}

#[derive(Clone)]
pub struct List<T> {
    head: Option<Rc<Node<T>>>,
    len: usize,
}

impl<T> List<T> {
    fn new() -> List<T> {
        List { head: None, len: 0 }
    }

    /// Returns a new list starting with the given value.
    fn add(&self, value: T) -> List<T> {
        List {
            head: Some(Rc::new(Node { value, next: self.head.clone() })),
            len: self.len + 1,
        }
    }

    /// Splits a list into the head and a list of the nodes that follow it.
    fn split(&self) -> (Option<&T>, List<T>) {
        if let Some(n) = self.head.as_ref() {
            (Some(&n.value), List { head: n.next.clone(), len: self.len - 1 })
        } else {
            (None, List { head: self.head.clone(), len: self.len })
        }
    }

    fn iter(&self) -> ListIter<T> {
        ListIter { node: self.head.as_ref() }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.head.is_none()
    }
}

impl<T: Eq + PartialEq> List<T> {
    fn contains(&self, value: &T) -> bool {
        self.iter().any(|v| v == value)
    }
}

impl<T: Clone + std::fmt::Debug> List<T> {
    /// Merges `self` and `other`.
    fn merge(&self, other: List<T>) -> List<T> {
        let mut new_list = List::new();

        for value in self.iter().chain(other.iter()) {
            new_list = new_list.add(value.clone());
        }

        new_list.rev()
    }

    /// Returns a new list with the values in reverse order.
    fn rev(&self) -> List<T> {
        let mut new_list = List::new();

        for v in self.iter() {
            new_list = new_list.add(v.clone());
        }

        new_list
    }
}

impl<T: fmt::Debug> fmt::Debug for List<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// An iterator over the values in an immutable list.
struct ListIter<'a, T> {
    node: Option<&'a Rc<Node<T>>>,
}

impl<'a, T> Iterator for ListIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.node.take() {
            self.node = node.next.as_ref();

            Some(&node.value)
        } else {
            None
        }
    }
}

/// The `con` (= constructor) type in the paper.
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
pub struct Con {
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
    Cons(Con, List<Pattern>),
    Var(String),
}

/// The `termd` type from the paper.
#[derive(Debug, Clone)]
pub enum TermDesc {
    // `Cons` is the top-most constructor, and its components are described by
    // the Vec.
    Pos(Con, List<TermDesc>),

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
    Neg(List<Con>),
}

/// The `access` type in the paper.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Access {
    Obj,
    Sel(usize, Box<Access>),
}

/// The `decision` type in the paper.
#[derive(Debug, Eq, PartialEq)]
pub enum Decision {
    /// A pattern didn't match.
    Failure,

    /// A pattern is matched and the right-hand value is to be returned.
    Success(RHS),

    /// Perform a runtime match of some sort.
    ///
    /// The members are as follows:
    ///
    /// 1. The value to test
    /// 2. The pattern/value to match against
    /// 3. The path to take upon a match
    /// 4. The path to take upon a failure.
    IfEq(Access, Con, Box<Decision>, Box<Decision>),
}

/// The result of the `staticmatch` (or in our case `static_match`) function.
#[derive(Debug)]
enum StaticMatch {
    Yes,
    No,
    Maybe,
}

/// `type context = (con * termd list) list` in the paper.
type Context = List<(Con, List<TermDesc>)>;

/// The work stack as used in the paper.
///
/// The paper uses a list of triple lists, removing the need for some append
/// operations. This is a bit annoying to work with in Rust (we have to unwrap()
/// in some places), but again we're trying to stay as close to the paper as
/// possible.
///
/// We use a type alias here so we don't have to re-type this type name in the
/// various places that it's used.
type Work = List<(List<Pattern>, List<Access>, List<TermDesc>)>;

/// The type of the right-hand side of a case (i.e. the code to run).
///
/// For the sake of simplicity we just use a String here. In a real compiler
/// this would probably be an AST node or another sort of IR to run upon a
/// match.
pub type RHS = String;

/// The `addneg` function in the paper.
fn addneg(dsc: TermDesc, con: Con) -> TermDesc {
    match dsc {
        // The paper introduces this function as an inexhaustive function. The
        // implementation in the Moscow ML compiler just returns the term when
        // it's a Pos, so we do the same.
        TermDesc::Pos(_, _) => dsc,
        TermDesc::Neg(nonset) => TermDesc::Neg(nonset.add(con)),
    }
}

/// The `staticmatch` function in the paper.
fn staticmatch(pcon: &Con, term: &TermDesc) -> StaticMatch {
    match term {
        TermDesc::Pos(scon, _) => {
            if pcon == scon {
                StaticMatch::Yes
            } else {
                StaticMatch::No
            }
        }
        TermDesc::Neg(excluded) => {
            if excluded.contains(pcon) {
                StaticMatch::No
            } else if pcon.span == (excluded.len() + 1) {
                // The way this works is as follows:
                //
                // A boolean has a span of two, as it has two constructors (true
                // and false).
                //
                // The `if` above means we determined our constructor IS NOT in
                // the deny list. Due to static typing, our list can't
                // contain unrelated constructors (e.g. an ADT constructor).
                //
                // Thus, if the length of the deny list is one less than the
                // span of our type, we know for a fact our constructor matches
                // the remaining constructor.
                //
                // In other words: we know we are NOT A, B, and C, and the only
                // remaining option is D. Thus, we match D.
                StaticMatch::Yes
            } else {
                StaticMatch::Maybe
            }
        }
    }
}

/// The equivalent of `List.tabulate` as used in the paper.
///
/// This function is kind of pointless in Rust as we could just use map(), but
/// we try to stay as close to the paper as possible in this implementation.
fn tabulate<T, F: Fn(usize) -> T>(length: usize, func: F) -> List<T> {
    let mut list = List::new();

    for val in (0..length).rev() {
        list = list.add(func(val));
    }

    list
}

fn args<T, F: Fn(usize) -> T>(pcon: &Con, func: F) -> List<T> {
    tabulate(pcon.arity, func)
}

fn getdargs(pcon: &Con, term: TermDesc) -> List<TermDesc> {
    match term {
        TermDesc::Pos(_, dargs) => dargs,
        TermDesc::Neg(_) => {
            tabulate(pcon.arity, |_| TermDesc::Neg(List::new()))
        }
    }
}

fn getoargs(pcon: &Con, acc: Access) -> List<Access> {
    // The paper uses `i+1`, presumably because humans use "1" to address the
    // first element (or maybe this is an SML thing?). Unfortunately, this isn't
    // clarified in the paper. Since it doesn't seem to actually matter, and
    // basically everyting is 0-indexed, we drop the +1 here.
    args(pcon, |i| Access::Sel(i, Box::new(acc.clone())))
}

fn augment(ctx: Context, dsc: TermDesc) -> Context {
    let (val, rest) = ctx.split();

    if let Some((con, args)) = val {
        rest.add((con.clone(), args.add(dsc)))
    } else {
        rest
    }
}

fn norm(ctx: Context) -> Context {
    let (val, rest) = ctx.split();

    if let Some((con, args)) = val {
        augment(rest, TermDesc::Pos(con.clone(), args.rev()))
    } else {
        rest
    }
}

fn builddsc(ctx: Context, dsc: TermDesc, work: Work) -> TermDesc {
    if let (Some((con, args)), rest) = ctx.split() {
        let (job, workr) = work.split();
        let (_, _, dargs) = job.unwrap();

        // The paper uses the following code for this:
        //
        //     rev args @ (dsc :: dargs)
        //
        // SML parses this as follows:
        //
        //     (rev args)   @   (dsc :: dargs)
        //
        // That is: it first reverses `args`, then appends the result of
        // `(dsc :: dargs)` to it. If you were to _first_ merge the values and
        // then reverse, you'd get incorrect decision trees. Unfortunately, I
        // ran into exactly that bug, and it took me a few hours to figure out.
        // And this is why functions with arguments should use parentheses and
        // commas :)
        let new_dsc = TermDesc::Pos(
            con.clone(),
            args.rev().merge(dargs.add(dsc.clone())),
        );

        builddsc(rest, new_dsc, workr)
    } else {
        dsc
    }
}

fn fail(dsc: TermDesc, rules: List<(Pattern, RHS)>) -> Decision {
    if let (Some((pat1, rhs1)), rulesr) = rules.split() {
        matches(
            pat1.clone(),
            Access::Obj,
            dsc,
            List::new(),
            List::new(),
            rhs1.clone(),
            rulesr,
        )
    } else {
        Decision::Failure
    }
}

fn succeed(
    ctx: Context,
    work: Work,
    rhs: RHS,
    rules: List<(Pattern, RHS)>,
) -> Decision {
    if let (Some((pats, accs, dscs)), workr) = work.split() {
        if pats.is_empty() && accs.is_empty() && dscs.is_empty() {
            succeed(norm(ctx), workr, rhs, rules)
        } else {
            let (pat1, patr) = pats.split();
            let (obj1, objr) = accs.split();
            let (dsc1, dscr) = dscs.split();

            matches(
                pat1.unwrap().clone(),
                obj1.unwrap().clone(),
                dsc1.unwrap().clone(),
                ctx,
                workr.add((patr, objr, dscr)),
                rhs,
                rules,
            )
        }
    } else {
        Decision::Success(rhs)
    }
}

/// This corresponds to the inner function `succeed'` in the paper.
fn match_succeed(
    pcon: Con,
    pargs: List<Pattern>,
    obj: Access,
    dsc: TermDesc,
    ctx: Context,
    work: Work,
    rhs: RHS,
    rules: List<(Pattern, RHS)>,
) -> Decision {
    let oargs = getoargs(&pcon, obj);
    let dargs = getdargs(&pcon, dsc);

    succeed(
        ctx.add((pcon, List::new())),
        work.add((pargs, oargs, dargs)),
        rhs,
        rules,
    )
}

/// This corresponds to the inner function `fail'` in the paper.
fn match_fail(
    newdsc: TermDesc,
    ctx: Context,
    work: Work,
    rules: List<(Pattern, RHS)>,
) -> Decision {
    fail(builddsc(ctx, newdsc, work), rules)
}

fn matches(
    pat1: Pattern,
    obj: Access,
    dsc: TermDesc,
    ctx: Context,
    work: Work,
    rhs: RHS,
    rules: List<(Pattern, RHS)>,
) -> Decision {
    match pat1 {
        Pattern::Var(_) => succeed(augment(ctx, dsc), work, rhs, rules),
        Pattern::Cons(pcon, pargs) => match staticmatch(&pcon, &dsc) {
            StaticMatch::Yes => {
                match_succeed(pcon, pargs, obj, dsc, ctx, work, rhs, rules)
            }
            StaticMatch::No => match_fail(dsc, ctx, work, rules),
            StaticMatch::Maybe => {
                // In the paper the equivalent code makes two assumptions that
                // don't work in Rust:
                //
                // 1. Certain values can have multiple owners (e.g. the `dsc`
                //    value is shared between functions).
                // 2. When building the subtree for a matched value, the
                //    algorithm expects that variables such as `work` and
                //    `rules` _are not_ modified in place. If they are,
                //    generating the subtree for an unmatched value produces
                //    incorrect results.
                //
                // In case of shared ownership we just clone the values. In a
                // real compiler that probably wouldn't work very well, but for
                // the sake of this implementation it's good enough.
                Decision::IfEq(
                    obj.clone(),
                    pcon.clone(),
                    Box::new(match_succeed(
                        pcon.clone(),
                        pargs,
                        obj,
                        dsc.clone(),
                        ctx.clone(),
                        work.clone(),
                        rhs.clone(),
                        rules.clone(),
                    )),
                    Box::new(match_fail(addneg(dsc, pcon), ctx, work, rules)),
                )
            }
        },
    }
}

pub fn compile(rules: List<(Pattern, RHS)>) -> Decision {
    fail(TermDesc::Neg(List::new()), rules)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A macro for creating a linked list.
    ///
    /// Rust has no (linked) list literals, so we use this macro instead.
    /// Basically whenever you have the SML expression `[a; b; c]`, you'd
    /// instead use `list![a, b, c]`.
    ///
    /// When creating a list using this macro, the values are added to the end
    /// of the list.
    macro_rules! list {
    ($($value: expr),*$(,)?) => {{
        let temp = vec![$($value),*];
        let mut list = List::new();

        for val in temp.into_iter().rev() {
            list = list.add(val);
        }

        list
    }}
}

    fn con(name: &str, arity: usize, span: usize) -> Con {
        Con { name: name.to_string(), arity, span }
    }

    fn nil() -> Pattern {
        Pattern::Cons(con("nil", 0, 1), List::new())
    }

    fn tt() -> Pattern {
        Pattern::Cons(con("true", 0, 2), List::new())
    }

    fn ff() -> Pattern {
        Pattern::Cons(con("false", 0, 2), List::new())
    }

    fn pair(a: Pattern, b: Pattern) -> Pattern {
        Pattern::Cons(con("pair", 2, 1), list![a, b])
    }

    fn var(name: &str) -> Pattern {
        Pattern::Var(name.to_string())
    }

    fn if_eq(acc: Access, con: Con, ok: Decision, fail: Decision) -> Decision {
        Decision::IfEq(acc, con, Box::new(ok), Box::new(fail))
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

    #[test]
    fn test_list_push_pop() {
        let list1 = List::new();
        let list2 = list1.add(10);
        let list3 = list2.add(20);

        assert!(list1.head.is_none());
        assert!(list2.head.is_some());
        assert!(list3.head.is_some());

        assert_eq!(list2.split().0, Some(&10));
        assert_eq!(list2.split().0, Some(&10));
        assert_eq!(list3.split().0, Some(&20));
    }

    #[test]
    fn test_list_rev() {
        let list1 = list![3, 2, 1];
        let list2 = list1.rev();

        assert_eq!(list1.iter().collect::<Vec<_>>(), vec![&3, &2, &1]);
        assert_eq!(list2.iter().collect::<Vec<_>>(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_list_rev_and_merge() {
        let list1 = list![3, 2, 1];
        let list2 = list![4];
        let list3 = list1.rev().merge(list2.add(10));

        assert_eq!(list3.iter().collect::<Vec<_>>(), vec![&1, &2, &3, &10, &4]);
    }

    #[test]
    fn test_list_merge() {
        let list1 = list![1, 2];
        let list2 = list![3, 4];
        let list3 = list1.merge(list2);

        assert_eq!(list3.iter().collect::<Vec<_>>(), vec![&1, &2, &3, &4]);
    }

    #[test]
    fn test_tabulate() {
        let vals = tabulate(3, |v| v);

        assert_eq!(vals.iter().collect::<Vec<_>>(), vec![&0, &1, &2]);
    }

    #[test]
    fn test_args() {
        let con = con("box", 2, 1);
        let vals = args(&con, |v| v);

        assert_eq!(vals.iter().collect::<Vec<_>>(), vec![&0, &1]);
    }

    #[test]
    fn test_getdargs_with_pos_term() {
        let con = con("box", 2, 1);
        let term =
            TermDesc::Pos(con.clone(), list![TermDesc::Neg(List::new())]);
        let args = getdargs(&con, term);
        let arg = args.iter().next();

        assert!(matches!(arg, Some(TermDesc::Neg(_))));
    }

    #[test]
    fn test_getdargs_with_neg_term() {
        let con = con("box", 2, 1);
        let term = TermDesc::Neg(List::new());
        let args = getdargs(&con, term);
        let mut iter = args.iter();

        assert!(matches!(iter.next(), Some(TermDesc::Neg(_))));
        assert!(matches!(iter.next(), Some(TermDesc::Neg(_))));
    }

    #[test]
    fn test_getoargs() {
        let con = con("box", 2, 1);
        let acc = sel(42, obj());
        let args = getoargs(&con, acc);

        assert_eq!(
            args.iter().collect::<Vec<_>>(),
            vec![&sel(0, sel(42, obj())), &sel(1, sel(42, obj()))]
        );
    }

    #[test]
    fn test_match_always_succeeds() {
        let result = compile(list![(nil(), rhs("true"))]);

        assert_eq!(result, success("true"));
    }

    #[test]
    fn test_match_always_fails() {
        let result = compile(List::new());

        assert_eq!(result, failure());
    }

    #[test]
    fn test_match_single_pattern() {
        let result = compile(list![(tt(), rhs("true")), (ff(), rhs("false")),]);

        assert_eq!(
            result,
            if_eq(obj(), con("true", 0, 2), success("true"), success("false"))
        );
    }

    #[test]
    fn test_match_var() {
        let result = compile(list![(var("a"), rhs("true"))]);

        assert_eq!(result, success("true"));
    }

    #[test]
    fn test_match_multiple_patterns() {
        let result = compile(list![
            (tt(), rhs("true")),
            (ff(), rhs("false")),
            (tt(), rhs("redundant"))
        ]);

        // Redundant patterns are ignored on the decision tree. This is also how
        // you'd detect redundant patterns: you'd somehow mark every RHS when
        // you produce their Success nodes. Any RHS nodes that remain unmarked
        // are redundant.
        assert_eq!(
            result,
            if_eq(obj(), con("true", 0, 2), success("true"), success("false"))
        );
    }

    #[test]
    fn test_inexhaustive_match() {
        let result = compile(list![(tt(), rhs("true")),]);

        assert_eq!(
            result,
            if_eq(obj(), con("true", 0, 2), success("true"), failure())
        );
    }

    #[test]
    fn test_inexhaustive_match_from_paper() {
        let green = Pattern::Cons(con("green", 0, 3), List::new());
        let result = compile(list![
            (pair(tt(), green.clone()), rhs("111")),
            (pair(ff(), green.clone()), rhs("222")),
        ]);

        assert_eq!(
            result,
            if_eq(
                sel(0, obj()),
                con("true", 0, 2),
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
    }

    #[test]
    fn test_nested_match() {
        let result = compile(list![
            (pair(tt(), tt()), rhs("foo")),
            (pair(tt(), ff()), rhs("bar")),
            (pair(ff(), ff()), rhs("baz")),
            (pair(ff(), tt()), rhs("quix")),
        ]);

        assert_eq!(
            result,
            if_eq(
                sel(0, obj()),
                con("true", 0, 2),
                if_eq(
                    sel(1, obj()),
                    con("true", 0, 2),
                    success("foo"),
                    success("bar")
                ),
                if_eq(
                    sel(1, obj()),
                    con("false", 0, 2),
                    success("baz"),
                    success("quix")
                )
            )
        );
    }
}
