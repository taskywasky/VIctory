//! Split waves: a console wave64 compute program on a host whose waves
//! have 32 lanes (NVIDIA). Each console wave runs as two host waves, its
//! low and high half (console lanes 0-31 and 32-63); a cross-lane operation
//! takes each half's part over the host wave and combines the two through
//! group-shared memory between thread-group barriers (see
//! `lower::Lower::exchange`).
//!
//! A barrier must be reached by every thread of the group, so exchanges are
//! only possible where the whole group runs together: in a block no lane
//! branch decides whether (or how often) a lane reaches. [`convergent`]
//! finds those blocks; elsewhere each half works alone, over its own 32
//! lanes.

use crate::structure::{Atom, LaneExit, Structured};

/// Per block: whether every thread of the group that starts the program
/// reaches it, together and as often as the others.
///
/// A block is divergent when it lies in the region a divergent branch
/// controls: reachable from the branch before its immediate
/// post-dominator. That covers both sides of a lane `if` and the body of a
/// loop with a lane exit. A branch is divergent unless its condition is
/// the same in every thread of the group: with one console wave per group
/// (`one_wave`), a condition over wave-uniform values ([`Atom::Uniform`]);
/// with several waves, none (scalar values may differ between waves).
pub(crate) fn convergent(st: &Structured, one_wave: bool) -> Vec<bool> {
    let n = st.exits.len();
    let succs: Vec<Vec<usize>> = st
        .exits
        .iter()
        .map(|e| match *e {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch {
                taken, not_taken, ..
            } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => Vec::new(),
        })
        .collect();
    let ipdom = post_dominators(&succs);
    let mut out = vec![true; n];
    for (d, e) in st.exits.iter().enumerate() {
        let divergent = match *e {
            LaneExit::Branch { when, .. } => {
                !one_wave
                    || !st
                        .bools
                        .support(when)
                        .iter()
                        .all(|a| matches!(a, Atom::Uniform(_)))
            }
            // Where each lane jumps is its own.
            LaneExit::Indirect => true,
            _ => false,
        };
        if !divergent {
            continue;
        }
        let stop = ipdom[d];
        let mut work: Vec<usize> = succs[d].clone();
        let mut seen = vec![false; n];
        while let Some(b) = work.pop() {
            if Some(b) == stop || seen[b] {
                continue;
            }
            seen[b] = true;
            out[b] = false;
            work.extend(&succs[b]);
        }
    }
    out
}

/// Immediate post-dominator per block (`None`: the virtual exit every
/// block without successors leads to, or a block that never exits).
fn post_dominators(succs: &[Vec<usize>]) -> Vec<Option<usize>> {
    let n = succs.len();
    let words = (n + 1).div_ceil(64);
    let exit = n;
    let full = {
        let mut f = vec![u64::MAX; words];
        let extra = words * 64 - (n + 1);
        if extra > 0 {
            f[words - 1] >>= extra;
        }
        f
    };
    // pdom[b]: blocks on every path from b to the exit, b included.
    let mut pdom = vec![full.clone(); n];
    let set = |s: &mut Vec<u64>, b: usize| s[b / 64] |= 1 << (b % 64);
    let has = |s: &[u64], b: usize| s[b / 64] >> (b % 64) & 1 != 0;
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            let mut next = if succs[b].is_empty() {
                let mut s = vec![0; words];
                set(&mut s, exit);
                s
            } else {
                let mut s = full.clone();
                for &t in &succs[b] {
                    for (w, x) in s.iter_mut().zip(&pdom[t]) {
                        *w &= x;
                    }
                }
                s
            };
            set(&mut next, b);
            if next != pdom[b] {
                pdom[b] = next;
                changed = true;
            }
        }
    }
    // The immediate one: the strict post-dominator that every other strict
    // post-dominator post-dominates.
    (0..n)
        .map(|b| {
            if has(&pdom[b], exit) && pdom[b] == full {
                // Never exits: no useful post-dominator.
                return None;
            }
            let strict: Vec<usize> = (0..n).filter(|&c| c != b && has(&pdom[b], c)).collect();
            strict
                .iter()
                .copied()
                .find(|&c| strict.iter().all(|&o| o == c || has(&pdom[c], o)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::post_dominators;

    #[test]
    fn post_dominators_of_an_if_and_a_loop() {
        // 0 -> {1, 2} -> 3 -> {3, 4}; 4 ends.
        let succs = vec![vec![1, 2], vec![3], vec![3], vec![3, 4], vec![]];
        let ipdom = post_dominators(&succs);
        assert_eq!(ipdom, vec![Some(3), Some(3), Some(3), Some(4), None]);
    }
}
