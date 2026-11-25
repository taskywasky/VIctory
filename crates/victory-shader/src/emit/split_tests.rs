//! Split waves end to end: a synthetic GFX10 wave64 compute program of one
//! console wave whose results depend on all 64 lanes, lowered for a host
//! with 64-lane waves (`WaveSize(64)`) and for one with 32-lane waves
//! only (`WaveSize(32)`, each console wave split over two host waves, as
//! on NVIDIA), and run on the hardware GPU and WARP. Both must give the
//! console's results.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const VCC: u8 = 106;
const LITERAL: u8 = 255;
const FF1_B64: u8 = 0x14;

/// SOP1 `op sdst, src0`.
const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0`.
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0`.
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOPC `op vcc, src0, vsrc1`.
const fn vopc(op: u8, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | (op as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2`.
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, vaddr, s[srsrc:srsrc+3], 0 offen
/// offset:byte_off`.
const fn buffer_store_dword_offen(vdata: u8, vaddr: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

const fn int(n: u8) -> u16 {
    128 + n as u16
}

/// Results per lane, in this order, 64 dwords each.
const RESULTS: usize = 5;

/// The program: per lane (`v0` its thread id, one console wave of 64),
/// 0: lanes below 40 counted from a compare's mask (`s_bcnt1_i32_b64`);
/// 1: the lane index (`v_mbcnt_lo/hi` of all lanes);
/// 2: lane 45's `3 * id + 7` (`v_readlane_b32`, a high-half lane);
/// 3: the first lane's `id + 100` (`v_readfirstlane_b32`);
/// 4: the first lane above 37, from a mask of high lanes only (`s_ff1_i32_b64`).
fn program() -> Vec<u32> {
    let mut w = Vec::new();
    // s[4:5]: the table at heap byte 0; s[8:11]: the UAV.
    w.push(sop1(0x03, 5, 0x80));
    w.extend(s_load_dwordx4(8, 4));
    // v1 = id * 4: the lane's byte offset.
    w.push(vop2(0x1A, 1, int(2), 0));
    // 0: v_cmp_gt_u32 vcc, 40, v0; s_bcnt1_i32_b64 s12, vcc.
    w.push(vopc(0xC4, int(40), 0));
    w.push(sop1(0x10, 12, VCC));
    w.push(vop1(0x01, 2, 12));
    // 1: v_mbcnt_lo_u32_b32 v3, -1, 0; v_mbcnt_hi_u32_b32 v3, -1, v3.
    w.extend(vop3(0x365, 3, 193, int(0), 0));
    w.extend(vop3(0x366, 3, 193, vgpr(3), 0));
    // 2: v4 = 3 * id + 7; v_readlane_b32 s13, v4, 45.
    w.extend(vop3(0x143, 4, vgpr(0), int(3), int(7)));
    w.extend(vop3(0x360, 13, vgpr(4), int(45), 0));
    w.push(vop1(0x01, 4, 13));
    // 3: v5 = id + 100; v_readfirstlane_b32 s14, v5.
    w.push(vop2(0x25, 5, LITERAL as u16, 0));
    w.push(100);
    w.push(vop1(0x02, 14, vgpr(5)));
    w.push(vop1(0x01, 5, 14));
    // 4: v_cmp_lt_u32 vcc, 37, v0 (lanes 38-63); s_ff1_i32_b64 s15, vcc.
    w.push(vopc(0xC1, int(37), 0));
    w.push(sop1(FF1_B64, 15, VCC));
    w.push(vop1(0x01, 6, 15));
    for (k, v) in [2u8, 3, 4, 5, 6].into_iter().enumerate() {
        w.extend(buffer_store_dword_offen(v, 1, 8, 256 * k as u16));
    }
    w.push(S_ENDPGM);
    w
}

fn expected(lane: u32) -> [u32; RESULTS] {
    [40, lane, 3 * 45 + 7, 100, 38]
}

/// Lowers a wave64 program of 64 threads (root constants in s0-s3, the
/// UAV table in s4) for a host whose widest wave is `host_wave_max`.
fn lower_words(words: &[u32], host_wave_max: u32) -> Result<lower::Lowered, crate::ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
        host_wave_max,
    })
}

/// [`program`] lowered for a host whose widest wave is `host_wave_max`.
fn build(host_wave_max: u32) -> HostShader {
    let lowered = lower_words(&program(), host_wave_max).expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

#[test]
fn a_split_wave64_gives_the_consoles_lanes_masks_and_reads() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for host_wave_max in [64, 32] {
        let shader = build(host_wave_max);
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                eprintln!("no SM 6.6 {label} device; skipped");
                continue;
            };
            let (root, pso) = match host.compute_pipeline(&shader) {
                Ok(p) => p,
                // A device without the wave size: the other build covers it.
                Err(e) => {
                    eprintln!("{label}, {host_wave_max}-lane host: no pipeline ({e}); skipped");
                    continue;
                }
            };
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(64 * RESULTS as u32)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            for lane in 0..64 {
                let got: Vec<u32> = (0..RESULTS).map(|k| out[0][64 * k + lane]).collect();
                assert_eq!(
                    got,
                    expected(lane as u32),
                    "{label}, {host_wave_max}-lane host, lane {lane}"
                );
            }
            ran += 1;
        }
    }
    assert!(ran > 0, "no device ran the program");
}

/// Runs `shader` on the hardware GPU, if it makes the pipeline, and
/// returns the UAV's first `dwords`.
fn run_hardware(shader: &HostShader, dwords: u32) -> Option<Vec<u32>> {
    let host = Host::new(false).ok()?;
    let (root, pso) = match host.compute_pipeline(shader) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("hardware: no pipeline ({e}); skipped");
            return None;
        }
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(dwords)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch");
    Some(out[0].clone())
}

#[test]
fn a_split_lane_read_in_a_lane_branch_reads_within_its_half() {
    // Lanes below 40 read lane 33's `3 * id + 7`: the high half's lanes
    // 40-63 skip the branch, so no barrier can pair the halves there and
    // each half reads lane 33's place in itself (lane 1 in the low half).
    // (A lane outside the branch would read undefined on any host.)
    let mut w = Vec::new();
    w.push(sop1(0x03, 5, 0x80));
    w.extend(s_load_dwordx4(8, 4));
    w.push(vop2(0x1A, 1, int(2), 0));
    w.extend(vop3(0x143, 4, vgpr(0), int(3), int(7)));
    w.push(vopc(0xC4, int(40), 0));
    // s_and_saveexec_b64 s[20:21], vcc; s_cbranch_execz over 5 dwords.
    w.push(sop1(0x24, 20, VCC));
    w.push(0xBF88_0000 | 5);
    w.extend(vop3(0x360, 13, vgpr(4), int(33), 0));
    w.push(vop1(0x01, 2, 13));
    w.extend(buffer_store_dword_offen(2, 1, 8, 0));
    // s_or_b64 exec, exec, s[20:21].
    w.push(0x8000_0000 | 0x11 << 23 | 126 << 16 | 20 << 8 | 126);
    w.push(S_ENDPGM);
    assert!(lower_words(&w, 64).is_ok());
    let lowered = lower_words(&w, 32).expect("split lowers it");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some(out) = run_hardware(&shader, 64) else {
        return;
    };
    for lane in 0..64u32 {
        let want = match lane {
            0..32 => 3 + 7,
            32..40 => 3 * 33 + 7,
            _ => 0,
        };
        assert_eq!(out[lane as usize], want, "lane {lane}");
    }
}
