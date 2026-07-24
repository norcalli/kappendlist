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

### Fixed

- Soundness: reads no longer create `&mut Inner`, and element access is
  per-slot, so references handed out by `push`/`get` are never invalidated by
  subsequent operations. The test suite is clean under Miri (strict provenance,
  both Stacked Borrows and Tree Borrows).
- Chunk allocations are over-aligned so pointer tagging is sound for
  small-alignment element types; tagging uses provenance-preserving `map_addr`.
- `Drain` now drops un-yielded elements instead of leaking them.
- `Iter::size_hint` no longer underflows past the end of iteration.
