//! A growable list you can append to through a shared `&self` reference while
//! references to existing elements stay alive.
//!
//! Elements are stored in geometrically growing batches (16, 32, 64, … elements)
//! that are **never moved or reallocated** once allocated. Because element
//! storage is stable, a reference returned by [`push`](AppendList::push) (or
//! [`get`](AppendList::get)) remains valid across later pushes — the thing that a
//! plain `Vec` cannot promise, since growing a `Vec` may move its buffer.
//!
//! Indexing is O(1) and table-free: batch `b` holds `16 << b` elements starting
//! at index `16 * (2^b - 1)`, so an index maps to its batch with a
//! `leading_zeros` and two shifts, then to a slot with a single load of that
//! batch's base pointer.
//!
//! This crate is `#![no_std]` (it only needs `alloc`) and uses the global
//! allocator, so it honors a custom `#[global_allocator]`.
//!
//! # Variants
//!
//! Two type aliases are provided over the shared implementation:
//!
//! * [`AppendList<T>`] — [`push`](AppendList::push) returns `&T`. Supports
//!   [`get`](AppendList::get), [`iter`](AppendList::iter), and `Index`.
//! * [`AppendListMut<T>`] — [`push`](AppendListMut::push) returns `&mut T`.
//!   Each push hands back a unique `&mut` to the freshly inserted element.
//!
//! # Example
//!
//! ```
//! use kappendlist::AppendList;
//!
//! let list = AppendList::new();
//!
//! // Push through a shared reference and keep the returned borrow.
//! let first: &i32 = list.push(10);
//!
//! // Pushing more does not invalidate `first`.
//! list.push(20);
//! list.push(30);
//!
//! assert_eq!(*first, 10);
//! assert_eq!(list.len(), 3);
//! assert_eq!(list[2], 30);
//! assert_eq!(list.iter().copied().collect::<Vec<_>>(), vec![10, 20, 30]);
//! ```
//!
//! # What the borrow checker still prevents
//!
//! Appending through `&self` is allowed, but operations that need `&mut self`
//! (such as [`drain_all`](AppendList::drain_all)) are still blocked while any
//! element reference is alive:
//!
//! ```compile_fail
//! use kappendlist::AppendList;
//!
//! let mut list = AppendList::new();
//! let first = list.push(1);   // borrows `list` immutably
//! list.drain_all();           // ERROR: needs `&mut list`
//! assert_eq!(*first, 1);      // `first` is still used here
//! ```
//!
//! # Thread safety
//!
//! An [`AppendList`] is **not** `Sync`: appending through `&self` mutates shared
//! state without synchronization, so it must not be shared across threads. It is
//! `Send` when `T: Send` (the whole list may be moved to another thread). For a
//! thread-safe append-only vector, see the `boxcar` or `append-only-vec` crates.
//!
//! # Limitations
//!
//! Zero-sized types are not supported (there is nothing to allocate); using one
//! panics at monomorphization time.

#![no_std]
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

extern crate alloc;
#[cfg(test)]
#[macro_use]
extern crate std;

use alloc::alloc::{Layout, alloc as global_alloc, dealloc as global_dealloc, handle_alloc_error};
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::fmt::{self, Debug};
use core::iter::{FromIterator, FusedIterator};
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::{Index, IndexMut};

/// A list that can be appended to while its elements are borrowed.
///
/// You will normally use one of the [`AppendList`] or [`AppendListMut`] aliases
/// rather than naming this type directly. The `V` type parameter selects the
/// variant (see the [crate-level docs](crate)).
pub struct BaseAppendList<T, V> {
    inner: UnsafeCell<Inner<T>>,
    _variant: PhantomData<V>,
}

/// Marker types selecting the behaviour of [`BaseAppendList`].
pub mod variants {
    /// `push` hands back a unique `&mut` to the new element.
    pub struct PushMut;
    /// `push` hands back a shared `&`; enables `get`/`iter`/`Index`.
    pub struct Index;
}

/// An append-only list whose `push` returns `&T` and that supports `get`,
/// `iter`, and indexing with `[]`.
pub type AppendList<T> = BaseAppendList<T, variants::Index>;

/// An append-only list whose `push` returns a unique `&mut T` to the newly
/// inserted element.
pub type AppendListMut<T> = BaseAppendList<T, variants::PushMut>;

// The list owns its storage uniquely; moving it to another thread is sound when
// the elements are `Send`. It is deliberately *not* `Sync`: appending through
// `&self` is unsynchronized.
unsafe impl<T: Send, V> Send for BaseAppendList<T, V> {}

impl<T, V> Default for BaseAppendList<T, V> {
    #[inline]
    fn default() -> Self {
        Self {
            inner: UnsafeCell::new(Inner::default()),
            _variant: PhantomData,
        }
    }
}

impl<T: Clone, V> Clone for BaseAppendList<T, V> {
    fn clone(&self) -> Self {
        let out = Self::default();
        // SAFETY: the reborrow is dropped before this returns and does not
        // overlap any element reference (see `inner_mut`).
        let dst = unsafe { out.inner_mut() };
        let src = self.inner();
        // `Iter` walks whole batches at a time, which beats `get`-per-index.
        dst.extend(Iter::new(src).cloned());
        out
    }
}

impl<T, V> BaseAppendList<T, V> {
    /// Create a new, empty list. No allocation happens until the first push.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the inner state immutably.
    #[inline(always)]
    fn inner(&self) -> &Inner<T> {
        // SAFETY: read-only access; no `&mut Inner` is live for the returned
        // borrow, and this method never mutates through the cell.
        unsafe { &*self.inner.get() }
    }

    /// Borrow the inner state mutably through a shared `&self`.
    ///
    /// # Safety
    /// The caller must ensure no other reference to `Inner` (shared or unique)
    /// is live for the duration of the returned borrow. The `&mut Inner` only
    /// covers the `Inner` struct itself — element storage lives behind raw
    /// pointers in separately-allocated chunks, so references handed out into
    /// those chunks are *not* invalidated by this reborrow.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)] // interior mutability is the whole point
    unsafe fn inner_mut(&self) -> &mut Inner<T> {
        // SAFETY: forwarded to the caller (see the method's `# Safety`).
        unsafe { &mut *self.inner.get() }
    }

    /// Get a mutable reference to the item at `index`, if it is in bounds.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.inner.get_mut().get_mut(index)
    }

    /// Get an iterator yielding `&mut T` over every element.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        let end = self.len();
        IterMut {
            inner: self.inner.get(),
            index: 0,
            end,
            _marker: PhantomData,
        }
    }

    /// Move every element out of the list, leaving it empty (but keeping its
    /// allocated capacity).
    ///
    /// Elements not yielded before the [`Drain`] is dropped are dropped in
    /// place, exactly like [`Vec::drain`].
    #[inline]
    pub fn drain_all(&mut self) -> Drain<'_, T> {
        let inner = self.inner.get_mut();
        let len = inner.len;
        // Reset the length up-front so that if the `Drain` (or a panicking
        // element `Drop`) leaks, no element is dropped twice.
        inner.len = 0;
        inner.rewind_cursor();
        Drain {
            inner: inner as *mut Inner<T>,
            index: 0,
            len,
            _marker: PhantomData,
        }
    }

    /// Remove every element, dropping them in place while keeping capacity.
    #[inline]
    pub fn clear(&mut self) {
        // The returned `Drain` drops all remaining elements when it falls out
        // of scope here.
        self.drain_all();
    }

    /// The number of elements currently in the list.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner().len()
    }

    /// The number of elements the list can hold without allocating more chunks.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner().capacity()
    }

    /// Returns `true` if the list contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reserve capacity for at least `additional` more elements.
    ///
    /// Existing batches are never moved, so this only ever *adds* storage.
    /// Capacity always lands on a batch boundary (`16 * (2^n - 1)` elements), so
    /// this rounds up to the next one and may reserve up to twice what was asked
    /// for — the same growth schedule repeated pushes would have followed. Slots
    /// that are never pushed to are never written, so the rounding costs address
    /// space rather than resident memory.
    ///
    /// Takes `&self`, so it can be called while elements are borrowed.
    #[inline]
    pub fn reserve(&self, additional: usize) {
        // SAFETY: the reborrow is dropped before this returns and does not
        // overlap any element reference (see `inner_mut`).
        unsafe { self.inner_mut() }.reserve(additional)
    }

    /// Append every item of `iter` to the end of the list.
    ///
    /// Note that this takes `&self`, so it can be called while elements are
    /// borrowed.
    #[inline]
    pub fn extend<I: IntoIterator<Item = T>>(&self, iter: I) {
        // SAFETY: the reborrow is dropped before this returns and does not
        // overlap any element reference (see `inner_mut`).
        unsafe { self.inner_mut() }.extend(iter)
    }
}

impl<T> BaseAppendList<T, variants::PushMut> {
    /// Append an item and get a unique `&mut` to it.
    ///
    /// Takes `&self`, so it can be called while previously-returned references
    /// are alive. Each call inserts a *new* element, so the returned `&mut`
    /// never aliases a reference from an earlier push.
    #[inline]
    #[allow(clippy::mut_from_ref)] // each push yields a fresh, unaliased element
    pub fn push(&self, item: T) -> &mut T {
        // SAFETY: `push` writes a fresh slot and returns a reference into stable
        // chunk storage; the `&mut Inner` reborrow does not cover that storage.
        unsafe { self.inner_mut() }.push(item)
    }
}

impl<T> BaseAppendList<T, variants::Index> {
    /// Append an item and get a shared `&` to it.
    ///
    /// Takes `&self`, so it can be called while previously-returned references
    /// are alive.
    #[inline]
    pub fn push(&self, item: T) -> &T {
        // SAFETY: see `BaseAppendList::<_, PushMut>::push`. The returned `&mut`
        // is immediately reborrowed as `&`.
        unsafe { self.inner_mut() }.push(item)
    }

    /// Get a shared reference to the item at `index`, if it is in bounds.
    ///
    /// Returns `None` when `index` is out of bounds. You can also index with
    /// `[]`, which panics on out-of-bounds access instead.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner().get(index)
    }

    /// Get an iterator yielding `&T` over every element.
    ///
    /// The iterator re-reads the current length on each step, so appending to
    /// the list while iterating is sound and observes the newly pushed items.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self.inner.get())
    }
}

impl<T> Index<usize> for BaseAppendList<T, variants::Index> {
    type Output = T;

    #[inline]
    fn index(&self, idx: usize) -> &T {
        let inner = self.inner();
        if idx >= inner.len() {
            index_out_of_bounds(idx, inner.len());
        }
        // SAFETY: bounds-checked just above, so the slot exists and is
        // initialized; we form a `&` to that single slot only.
        unsafe { (*inner.slot(idx)).assume_init_ref() }
    }
}

impl<T> IndexMut<usize> for BaseAppendList<T, variants::Index> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut T {
        let inner = self.inner.get_mut();
        if idx >= inner.len() {
            index_out_of_bounds(idx, inner.len());
        }
        // SAFETY: as `index`, but a unique borrow of a single initialized slot.
        unsafe { (*inner.slot(idx)).assume_init_mut() }
    }
}

/// Out-of-line so the panic's arguments do not have to be kept live (and
/// spilled) across an indexing loop.
#[cold]
#[inline(never)]
fn index_out_of_bounds(idx: usize, len: usize) -> ! {
    panic!("index {idx} out of bounds for list of length {len}")
}

// ---------------------------------------------------------------------------
// Inner storage
// ---------------------------------------------------------------------------

struct Inner<T> {
    len: usize,
    /// Next free slot: where the following `push` writes. Equal to `end` when
    /// the batch holding element `len` is full or not yet allocated, which is
    /// the only case `push` has to leave its fast path for.
    next: *mut MaybeUninit<T>,
    /// One past the last slot of the batch that `next` points into.
    end: *mut MaybeUninit<T>,
    /// One entry per allocated batch, where batch `b` holds `CHUNK_SIZE << b`
    /// elements starting at element `batch_start(b)`. Entries are *biased*: each
    /// stores `base - batch_start(b)`, so the slot for an index is just
    /// `batches[batch_of(index)] + index` — one load and one add, with the
    /// per-batch offset folded in at allocation time. Only ~log2(len) entries,
    /// so this table stays cache-resident.
    batches: Vec<*mut MaybeUninit<T>>,
}

impl<T> Default for Inner<T> {
    #[inline]
    fn default() -> Self {
        Self {
            len: 0,
            next: core::ptr::null_mut(),
            end: core::ptr::null_mut(),
            batches: Vec::new(),
        }
    }
}

impl<T> Inner<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn capacity(&self) -> usize {
        batch_start(self.batches.len())
    }

    /// Raw pointer to the slot at `index`.
    ///
    /// # Safety
    /// The batch holding `index` must already exist (`index < capacity`).
    #[inline(always)]
    unsafe fn slot(&self, index: usize) -> *mut MaybeUninit<T> {
        // SAFETY: caller guarantees the batch exists; unbiasing lands inside it
        // by construction (`batch_of` is the inverse of `batch_start`).
        unsafe { (*self.batches.get_unchecked(batch_of(index))).wrapping_add(index) }
    }

    /// The run of contiguous slots starting at `index` and stopping at the end
    /// of its batch or at `end`, whichever comes first: a pointer to the first
    /// slot and the number of slots.
    ///
    /// # Safety
    /// `index < end` and `end <= capacity`.
    #[inline]
    unsafe fn run_from(&self, index: usize, end: usize) -> (*mut MaybeUninit<T>, usize) {
        let batch = batch_of(index);
        let batch_end = batch_start(batch) + batch_len(batch);
        // SAFETY: `index < end <= capacity`, so this batch is allocated.
        (
            unsafe { self.slot(index) },
            core::cmp::min(batch_end, end) - index,
        )
    }

    /// Real base pointer of batch `batch` (the biased entry, unbiased).
    ///
    /// # Safety
    /// `batch` must be an allocated batch.
    #[inline(always)]
    unsafe fn batch_base(&self, batch: usize) -> *mut MaybeUninit<T> {
        // SAFETY: caller guarantees the batch exists.
        unsafe { (*self.batches.get_unchecked(batch)).wrapping_add(batch_start(batch)) }
    }

    fn push(&mut self, item: T) -> &mut T {
        if self.next == self.end {
            self.advance_cursor();
        }

        // SAFETY: the cursor now points at a free, uninitialized slot inside an
        // allocated batch. We form a `&mut` to that single slot only, so no
        // reference to any other (already-initialized) slot is invalidated.
        unsafe {
            let slot = self.next;
            (*slot).write(item);
            self.next = slot.add(1);
            self.len += 1;
            (*slot).assume_init_mut()
        }
    }

    /// Move the push cursor to the batch holding element `len`, allocating that
    /// batch if it does not exist yet.
    ///
    /// Only reached when the current batch is exhausted, i.e. once per batch,
    /// so `len` is always exactly at a batch boundary here.
    #[cold]
    #[inline(never)]
    fn advance_cursor(&mut self) {
        let batch = batch_of(self.len);
        debug_assert_eq!(self.len, batch_start(batch));
        if batch == self.batches.len() {
            self.alloc_batch();
        }
        debug_assert!(batch < self.batches.len());
        // SAFETY: allocated just above if it did not already exist.
        let base = unsafe { self.batch_base(batch) };
        self.next = base;
        // SAFETY: `base` was allocated with room for `batch_len(batch)`
        // elements, so this is the one-past-the-end pointer of that allocation.
        self.end = unsafe { base.add(batch_len(batch)) };
    }

    /// Point the push cursor at element 0. Only valid when `len == 0`.
    fn rewind_cursor(&mut self) {
        debug_assert_eq!(self.len, 0);
        match self.batches.first() {
            // SAFETY: batch 0 starts at element 0, so its entry is unbiased,
            // and it holds `CHUNK_SIZE` elements.
            Some(&base) => {
                self.next = base;
                self.end = unsafe { base.add(batch_len(0)) };
            }
            None => {
                self.next = core::ptr::null_mut();
                self.end = core::ptr::null_mut();
            }
        }
    }

    fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        // SAFETY: `index < len` ⇒ the slot exists and is initialized. We form a
        // `&` to this single slot only.
        Some(unsafe { (*self.slot(index)).assume_init_ref() })
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        // SAFETY: as `get`, but a unique borrow of a single initialized slot.
        Some(unsafe { (*self.slot(index)).assume_init_mut() })
    }

    /// Move the initialized element at `index` out of the list.
    ///
    /// # Safety
    /// `index` must be in bounds and the slot must not be read again.
    #[inline(always)]
    unsafe fn take(&mut self, index: usize) -> T {
        // SAFETY: forwarded to the caller.
        unsafe { (*self.slot(index)).assume_init_read() }
    }

    /// Allocate the next batch (index `batches.len()`, holding
    /// `CHUNK_SIZE << batches.len()` elements).
    fn alloc_batch(&mut self) {
        assert_not_zst::<T>();
        let batch = self.batches.len();
        let layout = batch_layout::<T>(batch_len(batch));
        // SAFETY: `batch_len` is non-zero and `T` is not a ZST (checked in
        // `push`), so the layout has non-zero size.
        let base = unsafe { global_alloc(layout) } as *mut MaybeUninit<T>;
        if base.is_null() {
            handle_alloc_error(layout);
        }
        // Store it biased by the batch's first element index. Wrapping keeps the
        // allocation's provenance; every use adds the bias back before any
        // dereference, landing inside the allocation again.
        self.batches.push(base.wrapping_sub(batch_start(batch)));
    }

    /// Ensure there is room for `additional` more elements beyond `len`.
    ///
    /// Capacity only ever lands on a batch boundary, so this rounds up to the
    /// next one.
    fn reserve(&mut self, additional: usize) {
        let target_len = self
            .len
            .checked_add(additional)
            .expect("kappendlist capacity overflow");
        if target_len <= self.capacity() {
            return;
        }
        // Smallest batch count whose combined capacity covers `target_len`.
        // `target_len >= 1` here, since it exceeds a capacity of at least 0.
        let needed = batch_of(target_len - 1) + 1;
        while self.batches.len() < needed {
            self.alloc_batch();
        }
        // The cursor is left alone on purpose: if it was live it still points
        // at element `len`, and if it was exhausted the next `push` re-derives
        // it (and now finds the batch already allocated).
    }

    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for x in iter {
            self.push(x);
        }
    }
}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        // Drop the still-initialized elements `[0, len)`.
        for index in 0..self.len {
            // SAFETY: `index < len` ⇒ the slot is initialized and dropped once.
            unsafe { (*self.slot(index)).assume_init_drop() };
        }
        // Free the underlying batch allocations.
        for batch in 0..self.batches.len() {
            // SAFETY: unbiasing recovers exactly the pointer `alloc_batch` got
            // back (provenance included); it is freed exactly once, with the
            // layout it was allocated with.
            unsafe {
                let base = self.batch_base(batch);
                global_dealloc(base.cast(), batch_layout::<T>(batch_len(batch)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Batch geometry
// ---------------------------------------------------------------------------

/// Number of elements in the first batch. Must be a power of two. Batch `b`
/// holds `CHUNK_SIZE << b` elements, so `n` batches hold `CHUNK_SIZE * (2^n - 1)`
/// elements in total.
const CHUNK_SIZE: usize = 16;
const CHUNK_SHIFT: u32 = CHUNK_SIZE.trailing_zeros();

const _: () = assert!(CHUNK_SIZE.is_power_of_two());

/// Index of the batch holding element `index`.
///
/// `index` lives in batch `b` iff `2^b <= index / CHUNK_SIZE + 1 < 2^(b+1)`, so
/// `b` is just the position of that value's highest set bit.
#[inline(always)]
fn batch_of(index: usize) -> usize {
    let scaled = (index >> CHUNK_SHIFT) + 1;
    (usize::BITS - 1 - scaled.leading_zeros()) as usize
}

/// Index of the first element of batch `batch` — equivalently, the total
/// capacity of the `batch` batches before it.
#[inline(always)]
fn batch_start(batch: usize) -> usize {
    // `CHUNK_SIZE * (2^batch - 1)`, written to stay correct for `batch == 0`.
    (CHUNK_SIZE << batch) - CHUNK_SIZE
}

/// Number of elements in batch `batch`.
#[inline(always)]
fn batch_len(batch: usize) -> usize {
    CHUNK_SIZE << batch
}

/// Zero-sized types have no storage to point stable references at, and a batch
/// of them would be a zero-sized allocation. Every path that allocates storage
/// or measures a distance between slots calls this, so using a ZST fails at
/// monomorphization time rather than misbehaving at run time.
#[inline(always)]
fn assert_not_zst<T>() {
    const {
        assert!(
            core::mem::size_of::<T>() != 0,
            "kappendlist does not support zero-sized types"
        )
    };
}

/// Layout of a batch holding `elems` elements.
#[inline]
fn batch_layout<T>(elems: usize) -> Layout {
    Layout::array::<T>(elems).expect("kappendlist capacity overflow")
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl<T, V> Extend<T> for BaseAppendList<T, V> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        BaseAppendList::extend(self, iter)
    }
}

impl<T, V> FromIterator<T> for BaseAppendList<T, V> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let list = Self::default();
        list.extend(iter);
        list
    }
}

impl<'a, T> IntoIterator for &'a BaseAppendList<T, variants::Index> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, V> IntoIterator for BaseAppendList<T, V> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let mut inner = self.inner.into_inner();
        let len = inner.len;
        // The `IntoIter` takes over responsibility for dropping the elements, so
        // stop `Inner`'s own `Drop` from also dropping them.
        inner.len = 0;
        IntoIter {
            inner,
            index: 0,
            len,
        }
    }
}

impl<T: PartialEq> PartialEq for BaseAppendList<T, variants::Index> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Eq> Eq for BaseAppendList<T, variants::Index> {}

impl<T: Debug> Debug for BaseAppendList<T, variants::Index> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_list().entries(self.iter()).finish()
    }
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

/// Shared iterator returned by [`AppendList::iter`].
///
/// Holds a raw pointer to the list's inner state (rather than a `&Inner`) so
/// that appending to the list mid-iteration does not invalidate it. Because the
/// list may grow during iteration, this iterator is intentionally neither
/// [`ExactSizeIterator`] nor [`FusedIterator`].
///
/// `next` walks a run of contiguous slots (a batch, capped by the length
/// observed when the run was entered) with a pointer bump, and only re-reads the
/// list's length when it reaches the end of a run — which is what makes
/// appending mid-iteration observable while keeping the common step branch-light.
pub struct Iter<'a, T> {
    inner: *const Inner<T>,
    /// Next slot to yield; `null` before the first run is entered.
    ptr: *const MaybeUninit<T>,
    /// One past the last slot of the current run.
    limit: *const MaybeUninit<T>,
    /// Index one past the current run's last element. The current index is
    /// `run_end - (limit - ptr)`, so stepping costs a pointer bump and nothing
    /// else — the index is only reconstructed off the hot path.
    run_end: usize,
    _marker: PhantomData<&'a Inner<T>>,
}

impl<'a, T> Iter<'a, T> {
    #[inline]
    fn new(inner: *const Inner<T>) -> Self {
        Iter {
            inner,
            // Both null: the first `next` takes the run-refresh path, where
            // `ptr == limit` makes the starting index `run_end == 0`.
            ptr: core::ptr::null(),
            limit: core::ptr::null(),
            run_end: 0,
            _marker: PhantomData,
        }
    }

    /// Number of elements left in the current run.
    #[inline]
    fn left_in_run(&self) -> usize {
        assert_not_zst::<T>();
        (self.limit.addr() - self.ptr.addr()) / core::mem::size_of::<T>()
    }

    /// Index of the element `next` would yield.
    #[inline]
    fn index(&self) -> usize {
        self.run_end - self.left_in_run()
    }

    /// End of the current run: re-read the length and enter the run holding
    /// `index`, if there is one.
    #[cold]
    #[inline(never)]
    fn next_run(&mut self) -> Option<&'a T> {
        // SAFETY: `inner` is valid for `'a`; the transient `&Inner` is dropped
        // before returning.
        let inner = unsafe { &*self.inner };
        let len = inner.len;
        // `ptr == limit` here, so the current index is exactly `run_end`.
        let index = self.run_end;
        if index >= len {
            return None;
        }
        // SAFETY: `index < len <= capacity`, so the run is allocated, and every
        // slot in it is initialized (the run stops at `len`).
        unsafe {
            let (ptr, run_len) = inner.run_from(index, len);
            self.ptr = ptr.cast_const();
            self.limit = self.ptr.add(run_len);
            self.run_end = index + run_len;
            self.yield_next()
        }
    }

    /// Yield the slot at `ptr` and step past it.
    ///
    /// # Safety
    /// `ptr` must point at an initialized slot inside a live batch.
    #[inline(always)]
    unsafe fn yield_next(&mut self) -> Option<&'a T> {
        // SAFETY: the slot is initialized and lives in stable batch storage
        // that outlives `'a`, so the borrow may be extended to `'a`.
        let item = unsafe { &*self.ptr.cast::<T>() };
        // SAFETY: `ptr < limit`, so stepping stays within the allocation.
        self.ptr = unsafe { self.ptr.add(1) };
        Some(item)
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.ptr == self.limit {
            return self.next_run();
        }
        // SAFETY: `ptr < limit` ⇒ it points at an initialized slot of the run.
        unsafe { self.yield_next() }
    }

    /// Consumers built on `fold` (`sum`, `for_each`, `collect`, …) get to see
    /// each run as a plain slice, which the optimizer can unroll and vectorize
    /// the way it does for `Vec`. Runs are still re-derived from the current
    /// length between batches, so elements appended by `f` are still observed.
    #[inline]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;
        loop {
            if self.ptr == self.limit {
                // Between runs: re-read the length and enter the next one.
                match self.next_run() {
                    Some(item) => acc = f(acc, item),
                    None => return acc,
                }
            } else {
                // SAFETY: `ptr < limit`, so `ptr` is non-null and the run
                // `[ptr, limit)` is entirely initialized and stays valid: the
                // only thing `f` can do to the list is append, which writes to
                // slots past `limit`.
                let run: &'a [T] = unsafe {
                    core::slice::from_raw_parts(self.ptr.cast::<T>(), self.left_in_run())
                };
                // Consume the run up-front so `self` stays consistent for any
                // `next_run` after `f` appends.
                self.ptr = self.limit;
                for item in run {
                    acc = f(acc, item);
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // SAFETY: `inner` is valid for `'a`.
        let remaining = unsafe { &*self.inner }.len().saturating_sub(self.index());
        (remaining, Some(remaining))
    }
}

/// Mutable iterator returned by [`BaseAppendList::iter_mut`].
///
/// Created from `&mut self`, so the list cannot grow while it is alive; its
/// length is therefore fixed and it is a well-behaved exact-size, fused,
/// double-ended iterator.
pub struct IterMut<'a, T> {
    inner: *mut Inner<T>,
    index: usize,
    end: usize,
    _marker: PhantomData<&'a mut Inner<T>>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        if self.index >= self.end {
            return None;
        }
        // SAFETY: created from `&'a mut self`, so we have exclusive access for
        // `'a`. Each call yields a `&mut` to a distinct slot, so the extended
        // borrows never alias.
        let inner = unsafe { &mut *self.inner };
        let item = inner.get_mut(self.index).unwrap();
        self.index += 1;
        Some(unsafe { &mut *(item as *mut T) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.index;
        (remaining, Some(remaining))
    }

    /// See [`Iter::fold`]: consumers built on `fold` get whole runs as slices.
    #[inline]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;
        while self.index < self.end {
            // SAFETY: created from `&'a mut self`, so we have exclusive access
            // for `'a`, and `index < end <= len` makes the run initialized.
            // Runs are disjoint, so the extended borrows never alias.
            let run: &'a mut [T] = unsafe {
                let (ptr, run_len) = (*self.inner).run_from(self.index, self.end);
                core::slice::from_raw_parts_mut(ptr.cast::<T>(), run_len)
            };
            self.index += run.len();
            for item in run {
                acc = f(acc, item);
            }
        }
        acc
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut T> {
        if self.index >= self.end {
            return None;
        }
        self.end -= 1;
        // SAFETY: as `next`; `self.end` is a distinct, in-bounds index.
        let inner = unsafe { &mut *self.inner };
        let item = inner.get_mut(self.end).unwrap();
        Some(unsafe { &mut *(item as *mut T) })
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {}
impl<T> FusedIterator for IterMut<'_, T> {}

/// By-value draining iterator returned by [`BaseAppendList::drain_all`].
///
/// Any elements not yielded are dropped when the `Drain` is dropped.
pub struct Drain<'a, T> {
    inner: *mut Inner<T>,
    index: usize,
    len: usize,
    _marker: PhantomData<&'a mut Inner<T>>,
}

impl<T> Iterator for Drain<'_, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.index >= self.len {
            return None;
        }
        // SAFETY: `index < len <= original length`, so the slot is initialized
        // and has not been moved out yet.
        let item = unsafe { (*self.inner).take(self.index) };
        self.index += 1;
        Some(item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }

    /// Move the elements out a run at a time. The consumed count is bumped
    /// before each element is handed over, so a panic in `f` still leaves
    /// `Drop` with an accurate range to clean up.
    #[inline]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, T) -> B,
    {
        let mut acc = init;
        while self.index < self.len {
            // SAFETY: `index < len <= the length at construction`, so the whole
            // run is initialized and has not been moved out of yet.
            let (ptr, run_len) = unsafe { (*self.inner).run_from(self.index, self.len) };
            for i in 0..run_len {
                // SAFETY: `i < run_len`, so this slot is inside the run.
                let item = unsafe { ptr.add(i).read().assume_init() };
                self.index += 1;
                acc = f(acc, item);
            }
        }
        acc
    }
}

impl<T> DoubleEndedIterator for Drain<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.index >= self.len {
            return None;
        }
        self.len -= 1;
        // SAFETY: `len` now points at an initialized, not-yet-moved slot.
        Some(unsafe { (*self.inner).take(self.len) })
    }
}

impl<T> ExactSizeIterator for Drain<'_, T> {}
impl<T> FusedIterator for Drain<'_, T> {}

impl<T> Drop for Drain<'_, T> {
    fn drop(&mut self) {
        // Drop the elements that were never yielded.
        for index in self.index..self.len {
            // SAFETY: these slots are initialized and, having not been yielded,
            // not yet moved out.
            unsafe { (*self.inner).take(index) };
        }
    }
}

/// By-value consuming iterator returned by `into_iter` on an owned list.
pub struct IntoIter<T> {
    inner: Inner<T>,
    index: usize,
    len: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.index >= self.len {
            return None;
        }
        // SAFETY: `index < len`, so the slot is initialized and not yet moved.
        let item = unsafe { self.inner.take(self.index) };
        self.index += 1;
        Some(item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }

    /// Move the elements out a run at a time. The consumed count is bumped
    /// before each element is handed over, so a panic in `f` still leaves
    /// `Drop` with an accurate range to clean up.
    #[inline]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, T) -> B,
    {
        let mut acc = init;
        while self.index < self.len {
            // SAFETY: `index < len <= the length at construction`, so the whole
            // run is initialized and has not been moved out of yet.
            let (ptr, run_len) = unsafe { self.inner.run_from(self.index, self.len) };
            for i in 0..run_len {
                // SAFETY: `i < run_len`, so this slot is inside the run.
                let item = unsafe { ptr.add(i).read().assume_init() };
                self.index += 1;
                acc = f(acc, item);
            }
        }
        acc
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.index >= self.len {
            return None;
        }
        self.len -= 1;
        // SAFETY: `len` now points at an initialized, not-yet-moved slot.
        Some(unsafe { self.inner.take(self.len) })
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}
impl<T> FusedIterator for IntoIter<T> {}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        // Drop the elements that were never yielded. `self.inner.len` was set to
        // 0 at construction, so `Inner::drop` will only free the chunks.
        for index in self.index..self.len {
            // SAFETY: these slots are initialized and not yet moved out.
            unsafe { self.inner.take(index) };
        }
    }
}

// Compile the README's code examples as doctests without changing the rendered
// crate documentation. Keeps the README examples from rotting.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

#[cfg(test)]
mod test {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn from_iterator() {
        let l: AppendList<i32> = (0..100).collect();

        for i in 0..100 {
            assert_eq!(l[i], i as i32);
        }
    }

    #[test]
    fn iterator() {
        let l: AppendList<i32> = (0..100).collect();
        // Both `iter()` and `(&list).into_iter()` yield shared references.
        let mut i1 = l.iter();
        let mut i2 = (&l).into_iter();

        for item in 0..100 {
            assert_eq!(i1.next(), Some(&item));
            assert_eq!(i2.next(), Some(&item));
        }

        assert_eq!(i1.next(), None);
        assert_eq!(i2.next(), None);
    }

    #[test]
    fn equality() {
        let a = AppendList::new();
        let b = AppendList::new();

        assert_eq!(a, b);

        a.push("foo");

        assert_ne!(a, b);

        b.push("foo");

        assert_eq!(a, b);

        a.push("bar");
        a.push("baz");

        assert_ne!(a, b);
    }

    #[test]
    fn debug_format() {
        let l: AppendList<i32> = (0..3).collect();
        assert_eq!(format!("{l:?}"), "[0, 1, 2]");
    }

    #[test]
    fn clone_is_independent() {
        let a: AppendList<i32> = (0..50).collect();
        let b = a.clone();
        assert_eq!(a, b);
        a.push(999);
        assert_eq!(a.len(), 51);
        assert_eq!(b.len(), 50);
    }

    #[test]
    fn clear_keeps_capacity() {
        let mut l: AppendList<i32> = (0..100).collect();
        let cap = l.capacity();
        l.clear();
        assert!(l.is_empty());
        assert_eq!(l.capacity(), cap);
        l.push(7);
        assert_eq!(l[0], 7);
    }

    #[test]
    fn into_iter_owned() {
        let l: AppendList<i32> = (0..10).collect();
        assert_eq!(
            l.into_iter().collect::<Vec<_>>(),
            (0..10).collect::<Vec<_>>()
        );

        // Partial consumption still drops the rest exactly once.
        let counter = Rc::new(std::cell::Cell::new(0));
        struct Bomb(Rc<std::cell::Cell<usize>>);
        impl Drop for Bomb {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let l = AppendList::new();
        for _ in 0..10 {
            l.push(Bomb(counter.clone()));
        }
        let mut it = l.into_iter();
        drop(it.next());
        drop(it.next());
        drop(it);
        assert_eq!(counter.get(), 10);
    }

    #[test]
    fn double_ended_iters() {
        let mut l: AppendList<i32> = (0..6).collect();
        assert_eq!(
            l.iter_mut().rev().map(|x| *x).collect::<Vec<_>>(),
            vec![5, 4, 3, 2, 1, 0]
        );
        assert_eq!(
            l.into_iter().rev().collect::<Vec<_>>(),
            vec![5, 4, 3, 2, 1, 0]
        );

        let mut l: AppendList<i32> = (0..6).collect();
        let drained: Vec<i32> = l.drain_all().rev().collect();
        assert_eq!(drained, vec![5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn iterator_size_hint() {
        let l: AppendList<i32> = AppendList::new();
        let mut i = l.iter();
        assert_eq!(i.size_hint(), (0, Some(0)));

        l.push(1);
        assert_eq!(i.size_hint(), (1, Some(1)));

        l.push(2);
        assert_eq!(i.size_hint(), (2, Some(2)));

        i.next();
        assert_eq!(i.size_hint(), (1, Some(1)));

        l.push(3);
        assert_eq!(i.size_hint(), (2, Some(2)));

        i.next();
        assert_eq!(i.size_hint(), (1, Some(1)));

        i.next();
        assert_eq!(i.size_hint(), (0, Some(0)));

        // Calling `next` past the end must not underflow `size_hint`.
        assert_eq!(i.next(), None);
        assert_eq!(i.next(), None);
        assert_eq!(i.size_hint(), (0, Some(0)));
    }

    #[test]
    fn empty_list() {
        let n: AppendList<usize> = AppendList::new();

        assert_eq!(n.len(), 0);
        assert!(n.is_empty());
        assert_eq!(n.get(0), None);
        assert_eq!(n.capacity(), 0);
        assert_eq!(n.iter().next(), None);

        let d: AppendList<usize> = AppendList::default();

        assert_eq!(d.len(), 0);
        assert_eq!(d.get(0), None);
    }

    #[test]
    fn references_survive_pushes() {
        let l = AppendList::new();
        let mut refs = Vec::new();
        for i in 0..500 {
            refs.push(l.push(i));
        }
        for (i, r) in refs.iter().enumerate() {
            assert_eq!(**r, i);
        }
    }

    #[test]
    fn reserve_then_push_keeps_addresses() {
        let l = AppendList::new();
        l.reserve(1000);
        let cap = l.capacity();
        assert!(cap >= 1000);
        let first = l.push(1);
        // Pushing up to the reserved capacity must not reallocate.
        for i in 2..=cap {
            l.push(i);
        }
        assert_eq!(*first, 1);
        assert_eq!(l.capacity(), cap);
    }

    #[test]
    fn small_alignment_type() {
        // `u8` has alignment 1; the allocator must still hand back an
        // 8-aligned buffer (we over-align) so pointer tagging is sound.
        let l: AppendList<u8> = (0..200u16).map(|x| x as u8).collect();
        for i in 0..200usize {
            assert_eq!(*l.get(i).unwrap(), i as u8);
        }
        let mut l = l;
        let drained: Vec<u8> = l.drain_all().collect();
        assert_eq!(drained.len(), 200);
    }

    #[test]
    fn push_mut_variant() {
        let l = AppendListMut::new();
        let a = l.push(1);
        let b = l.push(2);
        *a += 10;
        *b += 20;
        assert_eq!(*a, 11);
        assert_eq!(*b, 22);

        let mut l = l;
        assert_eq!(l.iter_mut().map(|x| *x).collect::<Vec<_>>(), vec![11, 22]);
    }

    #[test]
    fn append_while_iterating() {
        let l: AppendList<i32> = (0..3).collect();
        let mut seen = Vec::new();
        let mut it = l.iter();
        seen.push(*it.next().unwrap());
        l.push(3);
        l.push(4);
        for x in it {
            seen.push(*x);
        }
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn append_while_iterating_across_chunks() {
        // Start with 20 elements (spanning chunk 0 and into chunk 1), consume
        // part of chunk 0, then append across several more chunk boundaries and
        // finish. Exercises the chunk-aware iterator's refresh path.
        let l: AppendList<usize> = (0..20).collect();
        let mut seen = Vec::new();
        let mut it = l.iter();
        for _ in 0..10 {
            seen.push(*it.next().unwrap());
        }
        for x in 20..100 {
            l.push(x);
        }
        for x in it {
            seen.push(*x);
        }
        assert_eq!(seen, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn fold_observes_appends_within_a_batch() {
        // `for_each` goes through `fold`, which walks whole runs as slices.
        // Element 17 lives in the same batch as the slots the pushes below
        // write to, so this covers appending *into the batch being walked*.
        let l: AppendList<usize> = (0..20).collect();
        let mut seen = Vec::new();
        l.iter().for_each(|&x| {
            seen.push(x);
            if x == 17 {
                for y in 20..30 {
                    l.push(y);
                }
            }
        });
        assert_eq!(seen, (0..30).collect::<Vec<_>>());
    }

    #[test]
    // The explicit `fold`s are the point: they exercise the overrides.
    #[allow(clippy::unnecessary_fold)]
    fn fold_matches_next_over_many_batches() {
        let l: AppendList<usize> = (0..1_000).collect();
        // `sum` uses `fold`; the manual loop uses `next`. They must agree.
        let via_fold: usize = l.iter().copied().sum();
        let mut via_next = 0;
        for &x in l.iter() {
            via_next += x;
        }
        assert_eq!(via_fold, via_next);
        assert_eq!(via_fold, (0..1_000).sum::<usize>());

        let mut m: AppendList<usize> = (0..1_000).collect();
        assert_eq!(m.iter_mut().fold(0, |a, x| a + *x), via_fold);
        assert_eq!(m.drain_all().fold(0, |a, x| a + x), via_fold);

        let o: AppendList<usize> = (0..1_000).collect();
        assert_eq!(o.into_iter().fold(0, |a, x| a + x), via_fold);
    }

    #[test]
    fn drain_fold_panic_drops_each_element_once() {
        use std::cell::Cell;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let counter = Rc::new(Cell::new(0));

        struct Bomb(Rc<Cell<usize>>);
        impl Drop for Bomb {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let mut l = AppendList::new();
        for _ in 0..100 {
            l.push(Bomb(counter.clone()));
        }

        // Panic partway through a folded drain: the elements already handed to
        // the closure are dropped by it, the rest by `Drain::drop`.
        let mut n = 0;
        let result = catch_unwind(AssertUnwindSafe(|| {
            l.drain_all().fold((), |(), bomb| {
                n += 1;
                if n == 40 {
                    panic!("boom");
                }
                drop(bomb);
            })
        }));

        assert!(result.is_err());
        assert_eq!(counter.get(), 100, "every element dropped exactly once");
        assert!(l.is_empty());
    }

    #[test]
    fn drain_partial_drops_remainder() {
        use std::cell::Cell;

        let counter = Rc::new(Cell::new(0));

        struct Bomb(Rc<Cell<usize>>);
        impl Drop for Bomb {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let mut l = AppendList::new();
        for _ in 0..10 {
            l.push(Bomb(counter.clone()));
        }

        {
            let mut d = l.drain_all();
            // Yield (and drop) two, then drop the Drain with 8 remaining.
            drop(d.next());
            drop(d.next());
        }

        assert_eq!(
            counter.get(),
            10,
            "all elements should be dropped exactly once"
        );
        assert_eq!(l.len(), 0);
        assert!(l.is_empty());
    }

    #[test]
    fn thousand_item_list() {
        test_big_list(1_025);
    }

    #[test]
    #[ignore = "slow; run explicitly with --ignored"]
    fn million_item_list() {
        test_big_list(1_000_000);
    }

    fn test_big_list(size: usize) {
        let l = AppendList::new();
        let mut refs: Vec<&usize> = Vec::new();

        assert!(l.inner().batches.is_empty());
        for i in 0..size {
            assert_eq!(l.len(), i);

            refs.push(l.push(i));
            assert_eq!(l.len(), i + 1);

            if size < 5_000 {
                check_batch_invariants(&l, l.len());
            }
        }

        // Every returned reference still points at the right element.
        for (i, &r) in refs.iter().enumerate() {
            assert_eq!(Some(r), l.get(i));
            assert_eq!(Some(r as *const _), l.get(i).map(|x| x as *const _));
        }

        let mut l = l;
        for (i, x) in l.drain_all().enumerate() {
            assert_eq!(x, i);
        }
        assert_eq!(l.len(), 0);
        assert!(l.is_empty());

        // Capacity is preserved by draining and matches the growth formula.
        let expected_chunks = chunks_to_reach(size.div_ceil(CHUNK_SIZE));
        assert_eq!(l.capacity(), expected_chunks * CHUNK_SIZE);
        assert_eq!(l.capacity() % CHUNK_SIZE, 0);
        // capacity / CHUNK_SIZE = 2^n - 1  =>  n = ceil_log2(size/CHUNK_SIZE + 1)
        assert_eq!(
            l.capacity() / CHUNK_SIZE,
            (1 << ceil_log2(size / CHUNK_SIZE + 1)) - 1
        );
        check_batch_invariants(&l, size);

        // The list is fully reusable after draining.
        l.push(1);
        assert_eq!(l.drain_all().collect::<Vec<_>>(), vec![1]);
    }

    /// Check the batch-geometry invariants for a list that has held (at its
    /// high water mark) `element_count` elements.
    fn check_batch_invariants(l: &AppendList<usize>, element_count: usize) {
        let inner = l.inner();
        // One batch per doubling: `n` batches hold 16 * (2^n - 1) elements.
        let expected_batches = if element_count == 0 {
            0
        } else {
            ceil_log2(element_count.div_ceil(CHUNK_SIZE))
        };
        assert_eq!(inner.batches.len(), expected_batches);
        assert_eq!(inner.capacity(), batch_start(inner.batches.len()));
        assert!(inner.capacity() >= element_count);
        // Every index maps back into an allocated batch that claims to hold it.
        for i in 0..element_count.min(200) {
            let batch = batch_of(i);
            assert!(batch < inner.batches.len());
            assert!(batch_start(batch) <= i);
            assert!(i < batch_start(batch) + batch_len(batch));
        }
    }

    /// Total chunks a pure doubling-growth list ends up with once it holds at
    /// least `wanted` chunks: the smallest `2^n - 1 >= wanted`.
    fn chunks_to_reach(wanted: usize) -> usize {
        let mut total = 0;
        while total < wanted {
            total = (total << 1) + 1;
        }
        total
    }

    /// `ceil(log2(x))` for `x >= 1`.
    fn ceil_log2(x: usize) -> usize {
        debug_assert!(x >= 1);
        (usize::BITS - x.leading_zeros()) as usize
    }
}
