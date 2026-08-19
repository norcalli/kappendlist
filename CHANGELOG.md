# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Initial (unpublished) `0.1.0`.

### Added

- `AppendList<T>` — an append-only list whose `push` takes `&self` and returns
  `&T`, with O(1) `get`/`Index` and `iter`. References to existing elements stay
  valid across later pushes.
- `AppendListMut<T>` — variant whose `push` returns a unique `&mut T`.
- `reserve`, `clear`, `Clone`, by-value `IntoIterator`, `IndexMut`, and
  `DoubleEndedIterator`/`ExactSizeIterator`/`FusedIterator` where the semantics
  are well-defined.
- `#![no_std]` (only requires `alloc`); uses the global allocator.

### Performance

- Elements live in geometrically growing batches (16, 32, 64, … elements)
  addressed by arithmetic instead of a per-chunk pointer table. That drops the
  table's memory (one pointer per 16 elements) and a dependent load from every
  access.
- `push` bumps a cached cursor rather than re-deriving the destination slot.
- `iter`/`iter_mut`/`drain_all`/`into_iter` walk a whole batch at a time, and
  their `fold` overrides hand each batch to the slice iterator, so `sum`,
  `for_each`, `collect` and other `fold`-based consumers vectorize.
- `Index`/`IndexMut` bounds-check directly instead of going through
  `Option::unwrap_or_else`, which had forced a spill on every indexed access.
- `reserve` now rounds capacity up to a batch boundary (`16 * (2^n - 1)`
  elements) — the same schedule repeated pushes follow — so it may reserve up to
  twice what was asked for. Untouched slots are never written, so this costs
  address space, not resident memory; dropping the pointer table more than pays
  for it (a 20M-element `u64` list peaks ~10% lower).

### Fixed

- Soundness: reads no longer create `&mut Inner`, and element access is
  per-slot, so references handed out by `push`/`get` are never invalidated by
  subsequent operations. The test suite is clean under Miri (strict provenance,
  both Stacked Borrows and Tree Borrows).
- `Drain` now drops un-yielded elements instead of leaking them.
- `Iter::size_hint` no longer underflows past the end of iteration.
