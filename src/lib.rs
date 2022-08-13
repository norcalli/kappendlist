#![allow(dead_code)]
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::UnsafeCell;
use std::fmt::{self, Debug};
use std::iter::FromIterator;
use std::mem::MaybeUninit;
use std::ops::Index;

// Must be a power of 2
const CHUNK_SIZE: usize = 16;
const CHUNK_MASK: usize = CHUNK_SIZE - 1;

/// A list that can be appended to while elements are borrowed
///
/// Uses pointer tagging to track allocated chunks of a fixed size.
///
/// Indexing is O(1)
pub struct BaseAppendList<T, V> {
    inner: UnsafeCell<Inner<T>>,
    _variant: std::marker::PhantomData<V>,
}

impl<T, V> Default for BaseAppendList<T, V> {
    #[inline]
    fn default() -> Self {
        Self {
            inner: Default::default(),
            _variant: std::marker::PhantomData,
        }
    }
}

pub type AppendList<T> = BaseAppendList<T, variants::Index>;
pub type AppendListMut<T> = BaseAppendList<T, variants::PushMut>;

fn chunks_needed_maintaining_invariant(total_chunk_count: usize) -> usize {
    // let initial_count = self.chunks.len();
    let initial_count = 0;
    let mut new_chunk_count = initial_count;

    // Need to allocate more chunks
    // In a geometric series, 2^(n+1) = Sum(2^n, 0, n) + 1
    // or generally, r^(n+1) = (r-1) * Sum(r^n, 0, n) + 1
    // So in order to double the number of previous chunks allocated,
    // allocate `len + 1` more.
    // So `new_len = len + len + 1 = len * 2 + 1 = (len << 1) + 1`
    // Could also probably do this with some kind of log2() but meh
    while new_chunk_count < total_chunk_count {
        new_chunk_count <<= 1;
        new_chunk_count += 1;
    }
    new_chunk_count - initial_count
}

pub mod variants {
    pub struct PushMut;
    pub struct Index;
}

impl<T, V> BaseAppendList<T, V> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get an item from the list, if it is in bounds
    ///
    /// Returns `None` if the `index` is out-of-bounds. Note that you can also
    /// index with `[]`, which will panic on out-of-bounds.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.inner.get_mut().get_mut(index)
    }

    ///// Get an item from the list, if it is in bounds
    /////
    ///// Returns `None` if the `index` is out-of-bounds. Note that you can also
    ///// index with `[]`, which will panic on out-of-bounds.
    //#[inline]
    //pub fn expand_and_get_mut(&self, index: usize) -> &mut T
    //where
    //    T: Default,
    //{
    //    unsafe { (&mut *self.inner.get()).expand_and_get_mut(index) }
    //}

    #[inline(always)]
    fn unsafe_inner(&self) -> &mut Inner<T> {
        unsafe { &mut *self.inner.get() }
    }

    /// Get an iterator over the list
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<T> {
        self.inner.get_mut().iter_mut()
    }

    #[inline]
    pub fn drain_all<'a>(&'a mut self) -> Drain<'a, T> {
        self.inner.get_mut().drain_all()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.unsafe_inner().len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.unsafe_inner().capacity()
        // self.unsafe_inner().chunks.len() * CHUNK_SIZE
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn extend<I: IntoIterator<Item = T>>(&self, iter: I) {
        self.unsafe_inner().extend(iter)
    }
}

impl<T> BaseAppendList<T, variants::PushMut> {
    /// Append an item to the end
    ///
    /// Note that this does not require `mut`.
    #[inline]
    pub fn push(&self, item: T) -> &mut T {
        self.unsafe_inner().push(item)
    }
}

impl<T> BaseAppendList<T, variants::Index> {
    /// Append an item to the end
    ///
    /// Note that this does not require `mut`.
    #[inline]
    pub fn push(&self, item: T) -> &T {
        self.unsafe_inner().push(item)
    }

    /// Get an item from the list, if it is in bounds
    ///
    /// Returns `None` if the `index` is out-of-bounds. Note that you can also
    /// index with `[]`, which will panic on out-of-bounds.
    #[inline]
    pub fn get<'a>(&'a self, index: usize) -> Option<&'a T> {
        unsafe { (&mut *self.inner.get()).get(index) }
    }

    /// Get an iterator over the list
    #[inline]
    pub fn iter(&self) -> Iter<T> {
        self.unsafe_inner().iter()
    }
}

impl<T> std::ops::Index<usize> for BaseAppendList<T, variants::Index> {
    type Output = T;

    fn index(&self, idx: usize) -> &T {
        self.get(idx).unwrap()
    }
}

struct Inner<T> {
    len: usize,
    chunks: Vec<Chunk<T, CHUNK_SIZE>>,
}

pub struct Chunk<T, const CHUNK_SIZE: usize>(*mut [MaybeUninit<T>; CHUNK_SIZE]);

impl<T, const CHUNK_SIZE: usize> Chunk<T, CHUNK_SIZE> {
    pub unsafe fn system_alloc(count: usize) -> impl Iterator<Item = Self> {
        Self::alloc(count, |layout| System.alloc(layout))
    }
    pub unsafe fn alloc(
        count: usize,
        alloc: impl FnOnce(Layout) -> *mut u8,
    ) -> impl Iterator<Item = Self> {
        // The memory layout of arrays is guaranteed to be compatible with
        // putting them next to eachother contiguously.
        // MaybeUninit has the same memory layout as T
        let first_chunk = alloc(Layout::array::<T>(CHUNK_SIZE * count).unwrap())
            as *mut [MaybeUninit<T>; CHUNK_SIZE];
        assert!(count <= isize::MAX as usize);
        std::iter::once(Chunk(tag_ptr(first_chunk, 1)))
            .chain((1..count).map(move |i| Chunk(first_chunk.offset(i as isize))))
    }
    #[inline]
    pub fn tag(&self) -> u64 {
        get_ptr_tag(self.0)
    }
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [MaybeUninit<T>; CHUNK_SIZE] {
        unsafe { &mut *self.ptr() }
    }
    #[inline]
    pub unsafe fn as_slice_mut_unsafe(&self) -> &mut [MaybeUninit<T>; CHUNK_SIZE] {
        &mut *self.ptr()
    }
    #[inline]
    pub fn as_slice(&self) -> &[MaybeUninit<T>; CHUNK_SIZE] {
        unsafe { &*self.ptr() }
    }
    #[inline]
    pub fn ptr(&self) -> *mut [MaybeUninit<T>; CHUNK_SIZE] {
        untagged(self.0)
    }

    #[inline]
    pub unsafe fn system_dealloc(chunks: &[Chunk<T, CHUNK_SIZE>]) {
        Self::dealloc(chunks, |ptr, layout| System.dealloc(ptr, layout))
    }

    pub unsafe fn dealloc(
        chunks: &[Chunk<T, CHUNK_SIZE>],
        mut dealloc: impl FnMut(*mut u8, Layout),
    ) {
        if chunks.is_empty() {
            return;
        }
        assert_eq!(
            chunks[0].tag(),
            1,
            "First chunk wasn't tagged with allocation"
        );
        let mut chunk_count = 0;
        for chunk in chunks.iter().rev() {
            chunk_count += 1;
            if chunk.tag() == 1 {
                let layout = Layout::array::<T>(CHUNK_SIZE * chunk_count).unwrap();
                dealloc(chunk.ptr().cast(), layout);
                chunk_count = 0;
            }
        }
        assert_eq!(
            chunk_count, 0,
            "Chunks which weren't terminated by a tagged chunk were found"
        );
    }
}

// impl<T> Drop for Chunk<T> {
//     fn drop(&mut self) {
//         if self.tag() == 1 {
//             unsafe {
//                 System.dealloc(self.0.cast(), layout);
//             }
//         }
//     }
// }

impl<T> Default for Inner<T> {
    fn default() -> Self {
        Self {
            len: 0,
            chunks: Default::default(),
        }
    }
}

pub const TAG_MASK: u64 = 0b111;
#[inline(always)]
pub fn modify_ptr<T>(p: *mut T, f: impl FnOnce(u64) -> u64) -> *mut T {
    f(p as u64) as *mut T
}
#[inline(always)]
pub fn get_ptr_tag<T>(p: *mut T) -> u64 {
    p as u64 & TAG_MASK
}
#[inline(always)]
pub fn untagged<T>(p: *mut T) -> *mut T {
    modify_ptr(p, |v| v & (!TAG_MASK))
}
#[inline(always)]
pub fn tag_ptr<T>(p: *mut T, tag: u64) -> *mut T {
    modify_ptr(p, |v| v | (tag & TAG_MASK))
}

impl<T> Inner<T> {
    /// In test builds, check all of the unsafe invariants
    ///
    /// In release builds, no-op
    fn check_invariants(&self) {
        // TODO?
        // #[cfg(test)]
        // {
        //     if self.len.get() > 0 {
        //     } else {
        //     }
        // }
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.chunks.len() * CHUNK_SIZE
    }

    #[inline(always)]
    unsafe fn get_chunk(&self, chunk_index: usize) -> &mut [MaybeUninit<T>; CHUNK_SIZE] {
        self.chunks.get_unchecked(chunk_index).as_slice_mut_unsafe()
    }

    /// Append an item to the end
    ///
    /// Note that this does not require `mut`.
    pub fn push(&mut self, item: T) -> &mut T {
        self.check_invariants();

        let new_index = self.len;
        let chunk_index = new_index / CHUNK_SIZE;
        let index_in_chunk = new_index & CHUNK_MASK;

        if chunk_index >= self.chunks.len() {
            // Need to allocate more chunks
            // In a geometric series, 2^(n+1) = Sum(2^n, 0, n) + 1
            // or generally, r^(n+1) = (r-1) * Sum(r^n, 0, n) + 1
            // So in order to double the number of previous chunks allocated,
            // allocate `len + 1` more.
            self.allocate_chunks(self.chunks.len() + 1);
        }
        debug_assert!(chunk_index < self.chunks.len());
        let chunk = unsafe { self.chunks.get_unchecked(chunk_index).as_slice_mut_unsafe() };
        chunk[index_in_chunk].write(item);

        self.len += 1;

        self.check_invariants();

        unsafe { chunk[index_in_chunk].assume_init_mut() }
    }

    fn allocate_chunks(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        self.chunks.extend(unsafe { Chunk::system_alloc(count) });
    }

    fn chunks_needed_maintaining_invariant(&self, total_chunk_count: usize) -> usize {
        let mut new_chunk_count = self.chunks.len();
        // Need to allocate more chunks
        // In a geometric series, 2^(n+1) = Sum(2^n, 0, n) + 1
        // or generally, r^(n+1) = (r-1) * Sum(r^n, 0, n) + 1
        // So in order to double the number of previous chunks allocated,
        // allocate `len + 1` more.
        // So `new_len = len + len + 1 = len * 2 + 1 = (len << 1) + 1`
        // Could also probably do this with some kind of log2() but meh
        while new_chunk_count < total_chunk_count {
            new_chunk_count <<= 1;
            new_chunk_count += 1;
        }
        new_chunk_count - self.chunks.len()
    }

    /// Get the length of the list
    pub fn len(&self) -> usize {
        self.check_invariants();
        self.len
    }

    /// Get an item from the list, if it is in bounds
    ///
    /// Returns `None` if the `index` is out-of-bounds. Note that you can also
    /// index with `[]`, which will panic on out-of-bounds.
    pub fn get<'a>(&'a self, index: usize) -> Option<&'a T> {
        self.check_invariants();

        if index >= self.len {
            return None;
        }
        let chunk_index = index / CHUNK_SIZE;
        let index_in_chunk = index & CHUNK_MASK;
        Some(unsafe { self.get_chunk(chunk_index)[index_in_chunk].assume_init_ref() })
    }

    /// Get an item from the list, if it is in bounds
    ///
    /// Returns `None` if the `index` is out-of-bounds. Note that you can also
    /// index with `[]`, which will panic on out-of-bounds.
    // pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
    pub fn get_mut<'a>(&'a mut self, index: usize) -> Option<&'a mut T> {
        self.check_invariants();
        if index >= self.len {
            return None;
        }
        let chunk_index = index / CHUNK_SIZE;
        let index_in_chunk = index & CHUNK_MASK;
        Some(unsafe { self.get_chunk(chunk_index)[index_in_chunk].assume_init_mut() })
    }

    /// Get an item from the list, if it is in bounds
    ///
    /// Returns `None` if the `index` is out-of-bounds. Note that you can also
    /// index with `[]`, which will panic on out-of-bounds.
    #[inline(always)]
    fn get_unchecked_move(&mut self, index: usize) -> T {
        let chunk_index = index / CHUNK_SIZE;
        let index_in_chunk = index & CHUNK_MASK;
        unsafe { self.get_chunk(chunk_index)[index_in_chunk].assume_init_read() }
    }

    /// Get an item from the list, if it is in bounds
    ///
    /// Returns `None` if the `index` is out-of-bounds. Note that you can also
    /// index with `[]`, which will panic on out-of-bounds.
    pub fn expand_and_get_mut(&mut self, index: usize) -> &mut T
    where
        T: Default,
    {
        self.check_invariants();
        if index >= self.len {
            if index >= self.capacity() {
                let min_chunks_needed = index / CHUNK_SIZE;
                self.allocate_chunks(self.chunks_needed_maintaining_invariant(min_chunks_needed));
            }
            self.extend(std::iter::repeat_with(Default::default).take(index + 1 - self.len));
        }
        assert!(index < self.capacity());
        let chunk_index = index / CHUNK_SIZE;
        let index_in_chunk = index & CHUNK_MASK;
        unsafe { self.get_chunk(chunk_index)[index_in_chunk].assume_init_mut() }
    }

    /// Get an iterator over the list
    pub fn iter<'a>(&'a self) -> Iter<'a, T> {
        self.check_invariants();
        Iter {
            list: self,
            index: 0,
        }
    }

    /// Get an iterator over the list
    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, T> {
        self.check_invariants();
        IterMut {
            list: self,
            index: 0,
        }
    }

    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        // TODO use size hint to reserve
        self.check_invariants();
        for x in iter {
            self.push(x);
        }
    }

    pub fn drain_all<'a>(&'a mut self) -> Drain<'a, T> {
        self.check_invariants();
        let len = self.len;
        self.len = 0;
        Drain {
            list: self,
            index: 0,
            len,
        }
    }
}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        let mut remaining = self.len;
        // Drop the individual elements
        // The last one will be truncated via .take() by counting how many are
        // lefts, and .take() works fine if it's greater than the number of
        // elements in the iterator.
        for chunk in self.chunks.iter_mut() {
            // Iterates at most CHUNK_SIZE elements.
            for elem in chunk.as_slice_mut().iter_mut().take(remaining) {
                unsafe {
                    elem.assume_init_drop();
                }
                remaining -= 1;
            }
        }
        // Deallocate the actual array chunks
        unsafe {
            Chunk::system_dealloc(self.chunks.as_slice());
        }
    }
}

#[inline]
pub const fn floor_log2(x: usize) -> usize {
    const BITS_PER_BYTE: usize = 8;

    BITS_PER_BYTE * std::mem::size_of::<usize>() - (x.leading_zeros() as usize) - 1
}

#[inline]
pub const fn ceil_log2(x: usize) -> usize {
    const BITS_PER_BYTE: usize = 8;

    BITS_PER_BYTE * std::mem::size_of::<usize>() - (x.leading_zeros() as usize)
}

impl<T> Index<usize> for Inner<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("Inner indexed beyond its length")
    }
}

impl<T, V> std::iter::Extend<T> for BaseAppendList<T, V> {
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

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: PartialEq> PartialEq for BaseAppendList<T, variants::Index> {
    fn eq(&self, other: &BaseAppendList<T, variants::Index>) -> bool {
        let mut s = self.iter();
        let mut o = other.iter();

        loop {
            match (s.next(), o.next()) {
                (Some(a), Some(b)) if a == b => {}
                (None, None) => return true,
                _ => return false,
            }
        }
    }
}

impl<T: Debug> Debug for BaseAppendList<T, variants::Index> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_list().entries(self.iter()).finish()
    }
}

pub struct Drain<'a, T> {
    list: &'a mut Inner<T>,
    index: usize,
    len: usize,
}

impl<T> Iterator for Drain<'_, T> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }
        Some(self.list.get_unchecked_move(self.index.post_increment()))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }
}

trait PostIncrement {
    fn post_increment(&mut self) -> Self;
}

impl PostIncrement for usize {
    #[inline(always)]
    fn post_increment(&mut self) -> Self {
        let res = *self;
        *self += 1;
        res
    }
}

pub struct Iter<'a, T> {
    list: &'a Inner<T>,
    index: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.list.get(self.index.post_increment())
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.list.len() - self.index;
        (remaining, Some(remaining))
    }
}

pub struct IterMut<'a, T> {
    list: &'a mut Inner<T>,
    index: usize,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline(always)]
    fn next<'b>(&'b mut self) -> Option<&'a mut T> {
        unsafe { &mut *(self.list as *mut Inner<T>) }.get_mut(self.index.post_increment())
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.list.len() - self.index;
        (remaining, Some(remaining))
    }
}

#[cfg(test)]
mod test {
    use super::*;

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
        let mut i1 = l.iter();
        let mut i2 = l.into_iter();

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
    }

    // #[test]
    // fn chunk_sizes_make_sense() {
    //     assert_eq!(chunk_size(0), FIRST_CHUNK_SIZE);

    //     let mut index = 0;

    //     for chunk in 0..20 {
    //         // Each chunk starts just after the previous one ends
    //         assert_eq!(chunk_start(chunk), index);
    //         index += chunk_size(chunk);
    //     }
    // }

    // #[test]
    // fn index_chunk_matches_up() {
    //     for index in 0..1_000_000 {
    //         let chunk_id = index_chunk(index);

    //         // Each index happens after its chunk start and before its chunk end
    //         assert!(index >= chunk_start(chunk_id));
    //         assert!(index < chunk_start(chunk_id) + chunk_size(chunk_id));
    //     }
    // }

    #[test]
    fn empty_list() {
        let n: AppendList<usize> = AppendList::new();

        assert_eq!(n.len(), 0);
        assert_eq!(n.get(0), None);

        let d: AppendList<usize> = AppendList::default();

        assert_eq!(d.len(), 0);
        assert_eq!(d.get(0), None);
    }

    #[test]
    fn thousand_item_list() {
        // test_big_list(1_000);
        // test_big_list(1_024);
        test_big_list(1_025);
        // test_big_list(0);
        // test_big_list(1);
        // test_big_list(15);
        // test_big_list(16);
    }

    #[test]
    #[ignore]
    fn million_item_list() {
        test_big_list(1_000_000);
    }

    fn test_big_list(size: usize) {
        let l = AppendList::new();
        let mut refs: Vec<&usize> = Vec::new();

        assert!(l.unsafe_inner().chunks.is_empty());
        for i in 0..size {
            assert_eq!(l.len(), i);

            refs.push(l.push(i));
            assert_eq!(l.len(), i + 1);

            // refs.push(&l[i]);
            if size < 5_000 {
                // The number of chunks makes sense.
                assert_eq!(
                    l.unsafe_inner().chunks.len(),
                    chunks_needed_maintaining_invariant((l.len() + CHUNK_SIZE - 1) / CHUNK_SIZE)
                );
                // The number of leading chunks with tag 1 is the log of the number of
                // chunks (based on how many times you'd have to reallocate)
                assert_eq!(
                    l.unsafe_inner()
                        .chunks
                        .iter()
                        .filter(|x| x.tag() == 1)
                        .count(),
                    ceil_log2(l.unsafe_inner().chunks.len())
                );
                // Tags are only 0 or 1
                assert_eq!(
                    l.unsafe_inner()
                        .chunks
                        .iter()
                        .filter(|x| match x.tag() {
                            0..=1 => false,
                            _ => true,
                        })
                        .count(),
                    0
                );
            }
        }

        for i in 0..size {
            assert_eq!(Some(refs[i]), l.get(i));
            assert_eq!(Some(refs[i] as *const _), l.get(i).map(|x| x as *const _));
        }
        let mut l = l;
        for (i, x) in l.drain_all().enumerate() {
            assert_eq!(x, i);
            // NOTE uncommenting this should fail
            // assert_eq!(x, *refs[i]);
        }
        assert_eq!(l.len(), 0);
        assert!(l.is_empty());
        assert_eq!(
            l.capacity(),
            chunks_needed_maintaining_invariant((size + CHUNK_SIZE - 1) / CHUNK_SIZE) * CHUNK_SIZE
        );
        assert_eq!(l.capacity() % CHUNK_SIZE, 0);
        // capacity = (2^n - 1) * CHUNK_SIZE
        // => capacity / CHUNK_SIZE = 2^n - 1
        // => log2(capacity / CHUNK_SIZE + 1) = n
        assert_eq!(
            l.capacity() / CHUNK_SIZE,
            (1 << ceil_log2(size / CHUNK_SIZE + 1)) - 1
        );
        // The number of chunks makes sense.
        assert_eq!(
            l.unsafe_inner().chunks.len(),
            chunks_needed_maintaining_invariant((size + CHUNK_SIZE - 1) / CHUNK_SIZE)
        );
        // The number of leading chunks with tag 1 is the log of the number of
        // chunks (based on how many times you'd have to reallocate)
        assert_eq!(
            l.unsafe_inner()
                .chunks
                .iter()
                .filter(|x| x.tag() == 1)
                .count(),
            ceil_log2(l.unsafe_inner().chunks.len())
        );
        // Tags are only 0 or 1
        assert_eq!(
            l.unsafe_inner()
                .chunks
                .iter()
                .filter(|x| match x.tag() {
                    0..=1 => false,
                    _ => true,
                })
                .count(),
            0
        );

        l.push(1);
        // {
        //     let x = l.push(1);
        //     let r = l.get(0).unwrap();
        //     assert_eq!(*r, 1);
        //     *x = 2;
        //     assert_eq!(*r, 2);
        //     *x = 1;
        //     assert_eq!(*r, 1);
        // }
        assert_eq!(l.drain_all().collect::<Vec<_>>(), vec![1]);
        // NOTE uncommenting this should fail
        // assert_eq!(*refs[0], 0);
    }
}
