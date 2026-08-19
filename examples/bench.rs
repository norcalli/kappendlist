//! Benchmark harness used while tuning the crate. Best-of-N wall clock.
//!
//! ```sh
//! cargo run --release --example bench [N]
//! ```
#![allow(clippy::needless_range_loop)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use kappendlist::AppendList;

const TRIALS: usize = 7;

fn best<T>(mut f: impl FnMut() -> T) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..TRIALS {
        let start = Instant::now();
        let out = f();
        let elapsed = start.elapsed();
        black_box(out);
        best = best.min(elapsed);
    }
    best
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let trials = TRIALS;
    eprintln!("# N = {n}, best of {trials}");

    let ns_per = |d: Duration| d.as_nanos() as f64 / n as f64;
    let row = |name: &str, d: Duration| println!("{name:<22} {:>8.3} ns/el", ns_per(d));

    // --- construction ---
    row(
        "vec_push_reserved",
        best(|| {
            let mut v = Vec::with_capacity(n);
            for i in 0..n as u64 {
                v.push(i);
            }
            v
        }),
    );
    row(
        "list_push_reserved",
        best(|| {
            let l = AppendList::new();
            l.reserve(n);
            for i in 0..n as u64 {
                l.push(i);
            }
            l
        }),
    );
    row(
        "vec_push_grow",
        best(|| {
            let mut v = Vec::new();
            for i in 0..n as u64 {
                v.push(i);
            }
            v
        }),
    );
    row(
        "list_push_grow",
        best(|| {
            let l = AppendList::new();
            for i in 0..n as u64 {
                l.push(i);
            }
            l
        }),
    );
    row(
        "list_extend",
        best(|| {
            let l: AppendList<u64> = AppendList::new();
            l.extend(0..n as u64);
            l
        }),
    );
    row(
        "list_from_iter",
        best(|| (0..n as u64).collect::<AppendList<u64>>()),
    );

    // --- reads ---
    let mut v = Vec::with_capacity(n);
    for i in 0..n as u64 {
        v.push(i);
    }
    let l = AppendList::new();
    l.reserve(n);
    for i in 0..n as u64 {
        l.push(i);
    }

    row("vec_iter_sum", best(|| v.iter().copied().sum::<u64>()));
    row("list_iter_sum", best(|| l.iter().copied().sum::<u64>()));
    row(
        "vec_index_sum",
        best(|| {
            let mut s = 0u64;
            for i in 0..n {
                s = s.wrapping_add(v[i]);
            }
            s
        }),
    );
    row(
        "list_index_sum",
        best(|| {
            let mut s = 0u64;
            for i in 0..n {
                s = s.wrapping_add(l[i]);
            }
            s
        }),
    );
    // Random-ish (strided) access to defeat the prefetcher.
    let stride = 4099usize;
    row(
        "vec_index_stride",
        best(|| {
            let mut s = 0u64;
            let mut i = 0usize;
            for _ in 0..n {
                s = s.wrapping_add(v[i]);
                i = (i + stride) % n;
            }
            s
        }),
    );
    row(
        "list_index_stride",
        best(|| {
            let mut s = 0u64;
            let mut i = 0usize;
            for _ in 0..n {
                s = s.wrapping_add(l[i]);
                i = (i + stride) % n;
            }
            s
        }),
    );

    // --- consumption ---
    row(
        "list_clone",
        best(|| {
            let c = l.clone();
            black_box(&c);
            c
        }),
    );
    row(
        "list_drain_sum",
        best(|| {
            let mut c: AppendList<u64> = AppendList::new();
            c.reserve(n);
            c.extend(0..n as u64);
            let start = Instant::now();
            let s: u64 = c.drain_all().sum();
            black_box((s, start));
            c
        }),
    );
    row(
        "list_into_iter_sum",
        best(|| {
            let c: AppendList<u64> = (0..n as u64).collect();
            c.into_iter().sum::<u64>()
        }),
    );
}
