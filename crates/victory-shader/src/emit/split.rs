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

use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::structure::{Atom, Bool, LaneExit, Structured};

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
/// Another access after a write needs a barrier unless the compiler can
/// prove that two stores touch different host dwords.
pub(super) fn unfenced_lds(
    p: &Program,
    order: &[usize],
    uniform: &[bool],
    access: impl Fn(&Inst) -> bool,
    write: impl Fn(&Inst) -> bool,
    independent: impl Fn(usize, usize, &Inst, &Inst) -> bool,
) -> Option<usize> {
    let mut seen = vec![usize::MAX; p.blocks.len()];
    for &root in order {
        if !uniform[root] {
            continue;
        }
        let mut work = p.blocks[root].exit.successors();
        let mut previous = None;
        let mut multiple = false;
        let mut written = false;
        while let Some(b) = work.pop() {
            if uniform[b] || seen[b] == root {
                continue;
            }
            seen[b] = root;
            for inst in p.blocks[b].insts.iter().filter(|i| access(i)) {
                let writes = write(inst);
                if (written || previous.is_some() && writes)
                    && !(writes && written && !multiple
                        && previous.is_some_and(|(block, first)| block == b && independent(root, b, first, inst)))
                {
                    return Some(inst.machine.offset);
                }
                written |= writes;
                multiple |= previous.is_some();
                previous = Some((b, inst));
            }
            work.extend(p.blocks[b].exit.successors());
        }
    }
    None
}

/// Two ordinary stores in the same short lane-prefix arm can use separate
/// LDS ranges without synchronizing. Include the full dwords used by the
/// host to implement sub-dword stores.
pub(super) fn independent_stores(
    p: &Program, st: &Structured, root: usize, block: usize, first: &Inst, second: &Inst, bytes: u32,
) -> bool {
    let proof = || {
        let LaneExit::Branch { when, taken, not_taken } = st.exits[root] else { return None };
        let Some((Atom::Lane(test), lo, hi)) = st.bools.node(when) else { return None };
        let join = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) if taken == block => not_taken,
            (Bool::TRUE, Bool::FALSE) if not_taken == block => taken,
            _ => return None,
        };
        if p.blocks[block].preds != [root] || !matches!(st.exits[block], LaneExit::Jump(t) if t == join) {
            return None;
        }
        let plain = |b: usize, value: Value| {
            p.blocks[b].insts.iter().zip(&st.guards[b])
                .find(|(i, guard)| guard.is_none() && i.writes.iter().any(|&(_, v)| v == value))
                .map(|(i, _)| i)
        };
        let operands = |inst: &Inst| {
            let Fields::Valu(v) = &inst.fields else { return None };
            let &[a, b] = v.src.as_slice() else { return None };
            (v.neg == 0 && v.abs == 0 && v.opsel == 0 && v.sdwa.is_none() && v.dpp.is_none()).then_some((a, b))
        };
        let value = |inst: &Inst, op| match op {
            Operand::Vector { reg, dwords: 1 } => inst.read(Reg::V(reg)),
            _ => None,
        };
        let imm = |op| match op {
            Operand::Const(c @ 128..=192) => Some(u32::from(c - 128)),
            Operand::Literal(v) => Some(v),
            _ => None,
        };
        // The comparison itself may narrow EXEC; only the lane-id
        // definitions must execute unconditionally in its block.
        let cmp = p.blocks[root].insts.iter().find(|i| i.writes.iter().any(|&(_, v)| v == test))?;
        if !matches!(cmp.name(), "v_cmpx_gt_i32" | "v_cmp_gt_u32") {
            return None;
        }
        let (limit, lane) = operands(cmp)?;
        let count = imm(limit)?;
        if !(1..=64).contains(&count) {
            return None;
        }
        let lane = value(cmp, lane)?;
        let high = plain(root, lane)?;
        let (mask, low) = operands(high)?;
        let low = plain(root, value(high, low)?)?;
        if high.name() != "v_mbcnt_hi_u32_b32" || mask != Operand::Const(193)
            || low.name() != "v_mbcnt_lo_u32_b32" || operands(low)? != (Operand::Const(193), Operand::Const(128))
        {
            return None;
        }
        let range = |i: &Inst| {
            let Fields::Ds { addr: Some(addr), offset0, offset1, gds: false, .. } = i.fields else { return None };
            let def = plain(block, value(i, addr)?)?;
            let (shift, source) = operands(def)?;
            let shift = imm(shift)?;
            if def.name() != "v_lshlrev_b32" || value(def, source)? != lane || shift > 5 {
                return None;
            }
            let (start, end, align) = match i.name() {
                "ds_write_b16" => (u32::from(offset0) | u32::from(offset1) << 8, 2, 2),
                "ds_write_b32" => (u32::from(offset0) | u32::from(offset1) << 8, 4, 4),
                "ds_write2_b32" if offset0.abs_diff(offset1) == 1 => (u32::from(offset0.min(offset1)) * 4, 8, 4),
                _ => return None,
            };
            let last = (start + ((count - 1) << shift) + end).next_multiple_of(4);
            (start % align == 0 && (1 << shift) >= end && last <= bytes).then_some((start & !3, last))
        };
        let (a, b) = range(first)?;
        let (c, d) = range(second)?;
        Some(b <= c || d <= a)
    };
    proof().unwrap_or(false)
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
