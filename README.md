# kappendlist

An append-only list you can push to through a shared `&self` reference, where
references to existing elements stay valid across later pushes.

```rust
use kappendlist::AppendList;

let list = AppendList::new();

let first: &i32 = list.push(10); // push takes &self and returns a borrow
list.push(20);
list.push(30);                   // pushing more does not invalidate `first`

assert_eq!(*first, 10);
assert_eq!(list[2], 30);
assert_eq!(list.len(), 3);
```

## Why

A `Vec<T>` cannot hand out a `&T` and then let you `push`: growing the vector may
reallocate its buffer and move every element, dangling the reference. The borrow
checker forbids it for exactly this reason.

`kappendlist` stores elements in geometrically growing batches (16, 32, 64, …
elements) that are **never moved or reallocated** once allocated. Because element
storage is stable, a reference returned by `push` (or `get`) stays valid for as
long as the list lives — even as you keep pushing.

Indexing is **O(1)** and needs no side table: batch `b` holds `16 << b` elements
starting at index `16 * (2^b - 1)`, so an index maps to its batch with a
`leading_zeros` and to a slot with a single load of that batch's base pointer.
Push is a pointer bump against a cached cursor, and `iter()` walks a batch at a
time — as a slice, so `sum`, `for_each`, `collect` and friends vectorize the way
they do for `Vec`.

## Variants

| Type              | `push` returns | Extra API                        |
| ----------------- | -------------- | -------------------------------- |
| `AppendList<T>`   | `&T`           | `get`, `iter`, `Index` (`[]`)    |
| `AppendListMut<T>`| `&mut T`       | (each push yields a unique `&mut`)|

Both are aliases over `BaseAppendList<T, V>`. Shared-`&self` methods
(`push`, `get`, `iter`, `extend`, `reserve`) let you keep borrows alive; the
`&mut self` methods (`get_mut`, `iter_mut`, `drain_all`, `clear`, `IndexMut`) are
available on both. `Clone`, `Debug`, `PartialEq`, `FromIterator`, and by-value
`IntoIterator` are implemented too.

## Guarantees & limitations

- **Stable addresses.** References from `push`/`get` survive later pushes.
- **`#![no_std]`.** Only requires `alloc`, and allocates via the global
  allocator, so a custom `#[global_allocator]` is honored.
- **`Send` when `T: Send`.** The whole list can be moved between threads.
- **Not `Sync`.** Appending through `&self` is unsynchronized, so a list must not
  be *shared* across threads. For a thread-safe append-only vector use
  [`boxcar`](https://crates.io/crates/boxcar) or
  [`append-only-vec`](https://crates.io/crates/append-only-vec).
- **No zero-sized types.** There is no storage to point stable references at;
  pushing a ZST panics at monomorphization time.

## Correctness

This crate contains `unsafe` code (interior mutability, manual allocation, and
raw-pointer element access). The test suite passes under
[Miri](https://github.com/rust-lang/miri) with **strict provenance**, under both
**Stacked Borrows** and **Tree Borrows**:

```sh
cargo +nightly miri test
MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test
```

Batch base pointers are stored biased by the batch's first element index, so a
slot is one load plus one add. The bias is applied with `wrapping_sub`/
`wrapping_add`, which preserves provenance: every dereference happens through a
pointer that has been unbiased back into its own allocation.

## How it compares

`kappendlist` overlaps heavily with existing, better-maintained crates. Reach for
those first unless you specifically need the niche below:

| Crate               | Thread-safe | `push` returns | Stores `T` inline | Notes                                   |
| ------------------- | ----------- | -------------- | ----------------- | --------------------------------------- |
| `boxcar`            | yes         | `usize` index  | yes               | Most popular concurrent append-only Vec |
| `append-only-vec`   | yes         | `usize` index  | yes               | Concurrent, index-based                 |
| `elsa::FrozenVec`   | single/sync | `()`           | needs `Box<T>`    | Canonical "push while borrowed" crate   |
| `appendlist`        | no          | `()`           | yes               | The design ancestor; unmaintained (2019)|
| `typed-arena`       | no          | `&mut T`       | yes               | `&mut` through `&self`, but no indexing  |
| **`kappendlist`**   | no          | `&T` / `&mut T`| yes               | Indexable **and** `&mut`-returning push  |

The one thing none of the mainstream list crates offer is `AppendListMut`: an
**indexable** list whose `push` hands back a `&mut T` at a stable address.
`typed-arena`/`bumpalo` give you `&mut` through `&self` but no `get(i)`/`iter`;
the append-list crates give you indexing but only `&T`.

If you don't need that combination, prefer `boxcar` (concurrent) or
`elsa`/`appendlist` (single-threaded).

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
