//! A growable list you can append to through a shared `&self` reference while
//! references to existing elements stay alive.
//!
//! Elements are stored in fixed-size chunks (16 elements each) that are
//! allocated in geometrically growing batches and **never moved or reallocated**
//! once created. Because element storage is stable, a reference returned by
//! [`push`](AppendList::push) (or [`get`](AppendList::get)) remains valid across
//! later pushes — the thing that a plain `Vec` cannot promise, since growing a
//! `Vec` may move its buffer.
//!
//! Indexing is O(1): the element at `index` lives at chunk `index / 16`, offset
//! `index % 16`.
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
//! Zero-sized types are not supported (there is nothing to allocate); attempting
//! to push a ZST panics at monomorphization time.

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

/// Number of elements per chunk. Must be a power of two.
const CHUNK_SIZE: usize = 16;
const CHUNK_MASK: usize = CHUNK_SIZE - 1;

const _: () = assert!(CHUNK_SIZE.is_power_of_two());

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
        dst.extend((0..src.len()).map(|i| src.get(i).unwrap().clone()));
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
    /// Existing chunks are never moved, so this only ever *adds* storage. Takes
    /// `&self`, so it can be called while elements are borrowed.
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
        Iter {
            inner: self.inner.get(),
            index: 0,
            _marker: PhantomData,
        }
    }
}

impl<T> Index<usize> for BaseAppendList<T, variants::Index> {
    type Output = T;

    #[inline]
    fn index(&self, idx: usize) -> &T {
        self.get(idx).unwrap_or_else(|| {
            panic!(
                "index {idx} out of bounds for list of length {}",
                self.len()
            )
        })
    }
}

impl<T> IndexMut<usize> for BaseAppendList<T, variants::Index> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut T {
        let len = self.len();
        self.get_mut(idx)
            .unwrap_or_else(|| panic!("index {idx} out of bounds for list of length {len}"))
    }
}

// ---------------------------------------------------------------------------
// Inner storage
// ---------------------------------------------------------------------------

struct Inner<T> {
    len: usize,
    chunks: Vec<Chunk<T>>,
}

impl<T> Default for Inner<T> {
    #[inline]
    fn default() -> Self {
        Self {
            len: 0,
            chunks: Vec::new(),
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
        self.chunks.len() * CHUNK_SIZE
    }

    /// Raw pointer to the slot at `index`.
    ///
    /// # Safety
    /// The chunk holding `index` must already exist (`index < capacity`).
    #[inline(always)]
    unsafe fn slot(&self, index: usize) -> *mut MaybeUninit<T> {
        let chunk_index = index / CHUNK_SIZE;
        let index_in_chunk = index & CHUNK_MASK;
        // SAFETY: caller guarantees the chunk exists; `index_in_chunk < CHUNK_SIZE`.
        unsafe { self.chunks.get_unchecked(chunk_index).slot(index_in_chunk) }
    }

    fn push(&mut self, item: T) -> &mut T {
        // Zero-sized types have no storage to point stable references at.
        const {
            assert!(
                core::mem::size_of::<T>() != 0,
                "kappendlist does not support zero-sized types"
            )
        };

        let index = self.len;
        if index / CHUNK_SIZE >= self.chunks.len() {
            // Double the number of chunks (allocate `len + 1` more). See
            // `Chunk::alloc_batch` and the growth notes there.
            self.allocate_chunks(self.chunks.len() + 1);
        }

        // SAFETY: the chunk for `index` now exists; we form a `&mut` to this
        // single, previously-uninitialized slot only, so no reference to any
        // other (already-initialized) slot is invalidated.
        unsafe {
            let slot = self.slot(index);
            (*slot).write(item);
            self.len += 1;
            (*slot).assume_init_mut()
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

    fn allocate_chunks(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        // SAFETY: `count >= 1`, so `Chunk::alloc_batch` allocates a non-zero
        // layout.
        self.chunks.extend(unsafe { Chunk::alloc_batch(count) });
    }

    /// Ensure there is room for `additional` more elements beyond `len`.
    fn reserve(&mut self, additional: usize) {
        let target_len = self
            .len
            .checked_add(additional)
            .expect("kappendlist capacity overflow");
        let needed_chunks = target_len.div_ceil(CHUNK_SIZE);
        if needed_chunks > self.chunks.len() {
            // A single contiguous batch of exactly the shortfall; the tag scheme
            // handles batches of any size.
            self.allocate_chunks(needed_chunks - self.chunks.len());
        }
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
        // Free the underlying chunk allocations.
        // SAFETY: chunks were produced by `Chunk::alloc_batch` and are freed
        // exactly once here.
        unsafe { Chunk::dealloc_all(&self.chunks) };
    }
}

// ---------------------------------------------------------------------------
// Chunk: a tagged pointer to a CHUNK_SIZE-element buffer
// ---------------------------------------------------------------------------

/// Alignment we steal the low bits of the chunk pointer below. We request at
/// least this alignment from the allocator so tagging never clobbers a real
/// address bit.
const TAG_ALIGN: usize = 8;
const TAG_MASK: usize = TAG_ALIGN - 1;

const _: () = assert!(TAG_ALIGN.is_power_of_two());

/// A pointer to one `CHUNK_SIZE`-element buffer of `MaybeUninit<T>`.
///
/// Chunks are allocated in contiguous batches. The pointer of the *first* chunk
/// of each batch has its low bit set (tag `1`); every other chunk is untagged
/// (tag `0`). This lets [`dealloc_all`](Chunk::dealloc_all) recover batch
/// boundaries — and thus the original allocation size — when freeing.
struct Chunk<T> {
    /// Possibly-tagged pointer to the first element of this chunk's buffer.
    ptr: *mut MaybeUninit<T>,
}

/// Layout for a batch of `count` contiguous chunks, over-aligned to
/// [`TAG_ALIGN`] so the low bits are free for tagging.
#[inline]
fn batch_layout<T>(count: usize) -> Layout {
    let elems = CHUNK_SIZE
        .checked_mul(count)
        .expect("kappendlist capacity overflow");
    Layout::array::<T>(elems)
        .expect("chunk layout size overflow")
        .align_to(TAG_ALIGN)
        .expect("chunk layout alignment overflow")
}

impl<T> Chunk<T> {
    /// Allocate `count` contiguous chunks and yield one [`Chunk`] handle each,
    /// the first of which carries the batch tag.
    ///
    /// # Safety
    /// `count` must be `>= 1` so the allocation layout has non-zero size.
    unsafe fn alloc_batch(count: usize) -> impl Iterator<Item = Chunk<T>> {
        debug_assert!(count >= 1);
        let layout = batch_layout::<T>(count);
        // SAFETY: `count >= 1` and `T` is not a ZST (checked in `push`), so the
        // layout has non-zero size.
        let base = unsafe { global_alloc(layout) } as *mut MaybeUninit<T>;
        if base.is_null() {
            handle_alloc_error(layout);
        }
        debug_assert_eq!(
            addr_tag(base),
            0,
            "allocator returned an under-aligned pointer; tagging would corrupt it",
        );
        // The first chunk holds the tagged pointer; the rest are plain offsets
        // (by whole chunks) into the same allocation.
        core::iter::once(Chunk {
            ptr: with_tag(base, 1),
        })
        .chain((1..count).map(move |i| Chunk {
            // SAFETY: `i < count`, so `i * CHUNK_SIZE` stays within the batch
            // allocation of `count * CHUNK_SIZE` elements.
            ptr: unsafe { base.add(i * CHUNK_SIZE) },
        }))
    }

    /// The tag bits of this chunk's pointer (`1` for a batch head, else `0`).
    #[inline(always)]
    fn tag(&self) -> usize {
        addr_tag(self.ptr)
    }

    /// The real (untagged) base pointer of this chunk's buffer.
    #[inline(always)]
    fn base(&self) -> *mut MaybeUninit<T> {
        untag(self.ptr)
    }

    /// Raw pointer to element `i` within this chunk.
    ///
    /// # Safety
    /// `i` must be `< CHUNK_SIZE`.
    #[inline(always)]
    unsafe fn slot(&self, i: usize) -> *mut MaybeUninit<T> {
        debug_assert!(i < CHUNK_SIZE);
        // SAFETY: caller guarantees `i < CHUNK_SIZE`, within this chunk's buffer.
        unsafe { self.base().add(i) }
    }

    /// Free every batch backing `chunks`.
    ///
    /// # Safety
    /// `chunks` must be exactly the chunks produced by `alloc_batch` calls, in
    /// allocation order, and must not be freed again.
    unsafe fn dealloc_all(chunks: &[Chunk<T>]) {
        if chunks.is_empty() {
            return;
        }
        debug_assert_eq!(chunks[0].tag(), 1, "first chunk is not a batch head");
        // Walk backwards, counting chunks until we hit a batch head, then free
        // that whole batch with the layout it was allocated with.
        let mut count = 0;
        for chunk in chunks.iter().rev() {
            count += 1;
            if chunk.tag() == 1 {
                // SAFETY: `chunk` is a batch head allocated with this exact
                // layout by `alloc_batch`, and is freed exactly once.
                unsafe { global_dealloc(chunk.base().cast(), batch_layout::<T>(count)) };
                count = 0;
            }
        }
        debug_assert_eq!(count, 0, "trailing chunks without a batch head");
    }
}

// ---------------------------------------------------------------------------
// Tagged-pointer helpers (provenance-preserving)
// ---------------------------------------------------------------------------

/// The low tag bits of a pointer's address.
#[inline(always)]
fn addr_tag<T>(p: *mut T) -> usize {
    p.addr() & TAG_MASK
}

/// Clear the tag bits, preserving provenance.
#[inline(always)]
fn untag<T>(p: *mut T) -> *mut T {
    p.map_addr(|a| a & !TAG_MASK)
}

/// Set the given tag bits, preserving provenance.
#[inline(always)]
fn with_tag<T>(p: *mut T, tag: usize) -> *mut T {
    p.map_addr(|a| a | (tag & TAG_MASK))
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
pub struct Iter<'a, T> {
    inner: *const Inner<T>,
    index: usize,
    _marker: PhantomData<&'a Inner<T>>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        // SAFETY: `inner` is valid for `'a`; we form a transient `&Inner` and
        // extend the element borrow to `'a`, which is sound because the element
        // lives in stable chunk storage that outlives `'a`.
        let inner = unsafe { &*self.inner };
        let item = inner.get(self.index)?;
        self.index += 1;
        Some(unsafe { &*(item as *const T) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // SAFETY: `inner` is valid for `'a`.
        let remaining = unsafe { &*self.inner }.len().saturating_sub(self.index);
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

        assert!(l.inner().chunks.is_empty());
        for i in 0..size {
            assert_eq!(l.len(), i);

            refs.push(l.push(i));
            assert_eq!(l.len(), i + 1);

            if size < 5_000 {
                check_chunk_invariants(&l, l.len());
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
        check_chunk_invariants(&l, size);

        // The list is fully reusable after draining.
        l.push(1);
        assert_eq!(l.drain_all().collect::<Vec<_>>(), vec![1]);
    }

    /// Check the chunk-tagging invariants for a list that has held (at its high
    /// water mark) `element_count` elements via single pushes.
    fn check_chunk_invariants(l: &AppendList<usize>, element_count: usize) {
        let inner = l.inner();
        // The number of chunks matches the growth formula.
        assert_eq!(
            inner.chunks.len(),
            chunks_to_reach(element_count.div_ceil(CHUNK_SIZE))
        );
        // The number of batch-head chunks is log2 of the chunk count.
        assert_eq!(
            inner.chunks.iter().filter(|c| c.tag() == 1).count(),
            ceil_log2(inner.chunks.len())
        );
        // Tags are only ever 0 or 1.
        assert_eq!(inner.chunks.iter().filter(|c| c.tag() > 1).count(), 0);
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
