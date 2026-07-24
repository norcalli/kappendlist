//! Model-based (a.k.a. state-machine) property tests for `AppendList` /
//! `AppendListMut`.
//!
//! The main test (`model_matches_vec`) generates a random sequence of
//! operations and applies each one to both the system under test
//! (`AppendList<i32>`) and a plain `Vec<i32>` used as the oracle model. After
//! every single operation we assert the two are observably equivalent. If
//! `AppendList` ever disagrees with `Vec` about its contents, this test will
//! shrink the failing case down to a minimal operation sequence.
//!
//! In addition, a handful of standalone proptests exercise the specific
//! guarantees this crate advertises: reference stability across pushes, clone
//! independence, `into_iter` round-tripping (forwards and reversed), and the
//! `AppendListMut` "push returns `&mut T`" variant.

use kappendlist::{AppendList, AppendListMut};
use proptest::prelude::*;

/// Case count: kept small under Miri (which interprets every memory access
/// and is therefore orders of magnitude slower) so this test suite can still
/// run to completion there if invoked with `cargo miri test`.
fn case_count() -> u32 {
    if cfg!(miri) { 4 } else { 256 }
}

// ---------------------------------------------------------------------------
// Model-based test: random operation sequences vs. a `Vec` oracle.
// ---------------------------------------------------------------------------

/// A single operation applied to both the `AppendList` under test and the
/// `Vec` oracle model.
#[derive(Debug, Clone)]
enum Op {
    /// `list.push(x)` / `model.push(x)`.
    Push(i32),
    /// `list.extend(xs)` / `model.extend(xs)`.
    Extend(Vec<i32>),
    /// `list.reserve(n)`; must be a pure capacity hint with no observable
    /// effect on contents.
    Reserve(usize),
    /// `list.clear()` / `model.clear()`.
    Clear,
    /// `list.drain_all()`; must yield exactly the model's current contents
    /// (in order) and leave both empty.
    DrainAll,
    /// `list.get(i)` compared against `model.get(i)`.
    Get(usize),
}

/// A strategy generating arbitrary `Op` values.
fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        any::<i32>().prop_map(Op::Push),
        prop::collection::vec(any::<i32>(), 0..8).prop_map(Op::Extend),
        // Cap `Reserve` so it can't try to allocate a wildly large amount.
        (0usize..1000).prop_map(Op::Reserve),
        Just(Op::Clear),
        Just(Op::DrainAll),
        // Indices deliberately range a bit past any plausible model length so
        // out-of-bounds `Get`s (which should yield `None` on both sides) are
        // exercised too.
        (0usize..64).prop_map(Op::Get),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: case_count(),
        ..ProptestConfig::default()
    })]

    /// Apply a random sequence of `Op`s to an `AppendList<i32>` and a
    /// `Vec<i32>` in lockstep, checking full equivalence after every step.
    #[test]
    fn model_matches_vec(ops in prop::collection::vec(op_strategy(), 0..64)) {
        let mut list: AppendList<i32> = AppendList::new();
        let mut model: Vec<i32> = Vec::new();

        for op in ops {
            match op {
                Op::Push(x) => {
                    let r = list.push(x);
                    // The reference returned by `push` must observe the value
                    // just inserted.
                    prop_assert_eq!(*r, x);
                    model.push(x);
                }
                Op::Extend(xs) => {
                    list.extend(xs.clone());
                    model.extend(xs);
                }
                Op::Reserve(n) => {
                    // `reserve` is purely a capacity hint: it must not change
                    // anything observable about the list's contents.
                    list.reserve(n);
                }
                Op::Clear => {
                    list.clear();
                    model.clear();
                }
                Op::DrainAll => {
                    let drained: Vec<i32> = list.drain_all().collect();
                    prop_assert_eq!(&drained, &model);
                    model.clear();
                    prop_assert!(list.is_empty());
                }
                Op::Get(i) => {
                    prop_assert_eq!(list.get(i), model.get(i));
                }
            }

            // Invariants that must hold after *every* operation.
            prop_assert_eq!(list.len(), model.len());
            prop_assert_eq!(list.is_empty(), model.is_empty());
            // Check indexed access both within and just past the model's
            // length, including a couple of indices that are always
            // out-of-bounds.
            for i in 0..model.len() + 2 {
                prop_assert_eq!(list.get(i), model.get(i));
            }
            prop_assert_eq!(list.iter().copied().collect::<Vec<_>>(), model.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// Standalone property tests for specific guarantees.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig {
        cases: case_count(),
        ..ProptestConfig::default()
    })]

    /// The crate's core guarantee: pushing through `&self` never invalidates
    /// a previously returned reference. Push a random sequence of ints one at
    /// a time, keep every returned `&i32`, and check they all still read back
    /// the expected value at the end.
    #[test]
    fn references_survive_pushes(values in prop::collection::vec(any::<i32>(), 0..300)) {
        let list: AppendList<i32> = AppendList::new();
        let mut refs: Vec<&i32> = Vec::new();

        for &v in &values {
            refs.push(list.push(v));
        }

        prop_assert_eq!(refs.len(), values.len());
        for (r, &v) in refs.iter().zip(values.iter()) {
            prop_assert_eq!(**r, v);
        }
    }

    /// Cloning an `AppendList` must produce an independent copy: mutating the
    /// original afterwards (by pushing more elements) must not be observed
    /// through the clone.
    #[test]
    fn clone_is_independent(
        initial in prop::collection::vec(any::<i32>(), 0..50),
        extra in prop::collection::vec(any::<i32>(), 0..50),
    ) {
        let list: AppendList<i32> = initial.iter().copied().collect();
        let clone = list.clone();

        // The clone starts out equal to the original.
        prop_assert_eq!(&list, &clone);
        prop_assert_eq!(clone.iter().copied().collect::<Vec<_>>(), initial.clone());

        // Mutate the original only.
        for &x in &extra {
            list.push(x);
        }

        // The clone must be entirely unaffected...
        prop_assert_eq!(clone.len(), initial.len());
        prop_assert_eq!(clone.iter().copied().collect::<Vec<_>>(), initial.clone());

        // ...while the original reflects both the initial values and the
        // pushed-after-clone extras.
        let mut expected = initial;
        expected.extend(extra);
        prop_assert_eq!(list.iter().copied().collect::<Vec<_>>(), expected);
    }

    /// `into_iter()` on an owned `AppendList` must yield exactly the elements
    /// that were pushed, in order, and `.rev()` on it must yield them in
    /// reverse order (it's `DoubleEndedIterator`).
    #[test]
    fn into_iter_round_trip_and_reversed(values in prop::collection::vec(any::<i32>(), 0..200)) {
        let list: AppendList<i32> = values.iter().copied().collect();
        let collected: Vec<i32> = list.into_iter().collect();
        prop_assert_eq!(&collected, &values);

        let list_for_rev: AppendList<i32> = values.iter().copied().collect();
        let reversed: Vec<i32> = list_for_rev.into_iter().rev().collect();
        let mut expected_rev = values;
        expected_rev.reverse();
        prop_assert_eq!(reversed, expected_rev);
    }

    /// `AppendListMut::push` hands back a unique `&mut T` into the freshly
    /// inserted slot. Push a random `Vec`, write through every kept `&mut`
    /// reference, then confirm the writes stuck via `iter_mut`.
    #[test]
    fn append_list_mut_write_through(values in prop::collection::vec(any::<i32>(), 0..200)) {
        let list: AppendListMut<i32> = AppendListMut::new();

        // Scoped so every `&mut` borrow of `list` ends before we need to bind
        // `list` as `mut` and call `iter_mut` (which needs `&mut self`).
        {
            let mut refs: Vec<&mut i32> = Vec::new();
            for &v in &values {
                refs.push(list.push(v));
            }
            for r in refs.iter_mut() {
                **r += 1;
            }
        }

        let mut list = list;
        let via_iter_mut: Vec<i32> = list.iter_mut().map(|x| *x).collect();
        let expected: Vec<i32> = values.iter().map(|v| v + 1).collect();
        prop_assert_eq!(via_iter_mut, expected);
    }
}
