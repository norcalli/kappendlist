//! A rough overhead comparison of `AppendList<u64>` against `Vec<u64>`.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example overhead
//! ```
//!
//! This is a wall-clock microbenchmark (best-of-N to reduce noise), not a
//! retired-instruction count — treat the ratios as ballpark, not exact.

// The index-get benchmarks loop over indices on purpose: measuring `[i]` access
// is the point, so an iterator would defeat it.
#![allow(clippy::needless_range_loop)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use kappendlist::AppendList;

const N: usize = 20_000_000;
const TRIALS: usize = 7;

/// Run `f` `TRIALS` times and return the fastest observed duration.
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

fn ns_per(d: Duration) -> f64 {
    d.as_nanos() as f64 / N as f64
}

fn main() {
    println!("N = {N} elements, best of {TRIALS} trials\n");

    // ---- push (pre-reserved) ----
    let vec_push = best(|| {
        let mut v = Vec::with_capacity(N);
        for i in 0..N as u64 {
            v.push(i);
        }
        v
    });
    let list_push = best(|| {
        let l = AppendList::new();
        l.reserve(N);
        for i in 0..N as u64 {
            l.push(i);
        }
        l
    });

    // Build once, reuse for the read benchmarks.
    let mut v = Vec::with_capacity(N);
    for i in 0..N as u64 {
        v.push(i);
    }
    let l = AppendList::new();
    l.reserve(N);
    for i in 0..N as u64 {
        l.push(i);
    }

    // ---- sequential iteration ----
    let vec_iter = best(|| v.iter().copied().sum::<u64>());
    let list_iter = best(|| l.iter().copied().sum::<u64>());

    // ---- indexed get ----
    let vec_index = best(|| {
        let mut s = 0u64;
        for i in 0..N {
            s = s.wrapping_add(v[i]);
        }
        s
    });
    let list_index = best(|| {
        let mut s = 0u64;
        for i in 0..N {
            s = s.wrapping_add(l[i]);
        }
        s
    });

    let row = |name: &str, vec: Duration, list: Duration| {
        println!(
            "{name:<20} Vec {:>7.3} ns/el   AppendList {:>7.3} ns/el   ratio {:.2}x",
            ns_per(vec),
            ns_per(list),
            ns_per(list) / ns_per(vec),
        );
    };

    row("push (reserved)", vec_push, list_push);
    row("iterate (sum)", vec_iter, list_iter);
    row("index get (sum)", vec_index, list_index);
}
