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

use crate::structure::{LaneExit, Structured};

/// Per block: whether every thread of the group that starts the program
/// reaches it, together and as often as the others.
///
/// A scalar branch can differ between host halves after a half-local read.
/// Model watchdog returns as exits too: threads that return cannot reach
/// a later barrier, even if that block post-dominated the original loop.
pub(crate) fn convergent(
    st: &Structured,
    watchdog_return: impl Fn(usize, usize) -> bool,
) -> Vec<bool> {
    let exit = st.exits.len();
    let mut succs: Vec<Vec<usize>> = st
        .exits
        .iter()
        .enumerate()
        .map(|(b, e)| match *e {
            LaneExit::Jump(t) if watchdog_return(b, t) => vec![t, exit],
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch {
                taken, not_taken, ..
            } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => Vec::new(),
        })
        .collect();
    succs.push(Vec::new());
    let mut out = reconverged(&succs);
    out.pop();
    out
}

/// The two halves may run different arms of a region at the same time.
/// Multiple reads are safe; with a write we cannot order another access
/// without a group barrier, and cannot put one inside the region.
pub(crate) fn unfenced_lds(
    p: &crate::ir::Program,
    order: &[usize],
    uniform: &[bool],
    access: impl Fn(&crate::ir::Inst) -> bool,
    write: impl Fn(&crate::ir::Inst) -> bool,
) -> Option<usize> {
    let mut seen = vec![usize::MAX; p.blocks.len()];
    for &root in order {
        if !uniform[root] {
            continue;
        }
        let mut work = p.blocks[root].exit.successors();
        let (mut read, mut written) = (false, false);
        while let Some(b) = work.pop() {
            if uniform[b] || seen[b] == root {
                continue;
            }
            seen[b] = root;
            for inst in p.blocks[b].insts.iter().filter(|i| access(i)) {
                let writes = write(inst);
                if written || read && writes {
                    return Some(inst.machine.offset);
                }
                written |= writes;
                read |= !writes;
            }
            work.extend(p.blocks[b].exit.successors());
        }
    }
    None
}

fn reconverged(succs: &[Vec<usize>]) -> Vec<bool> {
    let n = succs.len();
    let ipdom = post_dominators(succs);
    let mut out = vec![true; n];
    for (d, edges) in succs.iter().enumerate() {
        if edges.len() < 2 {
            continue;
        }
        let stop = ipdom[d];
        let mut work = edges.clone();
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

    #[test]
    fn watchdog_return_prevents_reconvergence_after_the_loop() {
        // 0 hands control to a loop; 2 can take its watchdog exit at 4.
        let graph = [vec![1], vec![2, 3], vec![1, 4], vec![], vec![]];
        assert_eq!(super::reconverged(&graph), [true, false, false, false, false]);
    }
}
