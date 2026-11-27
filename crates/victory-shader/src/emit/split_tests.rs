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
fn lower_words(words: &[u32], host_wave_max: u32, lds_blocks: u32) -> Result<lower::Lowered, crate::ShaderError> {
    lower_words_with(words, host_wave_max, lds_blocks, &Default::default())
}

/// [`lower_words`] with [`structure::structure_with`]'s wave-level branches.
fn lower_words_with(
    words: &[u32],
    host_wave_max: u32,
    lds_blocks: u32,
    waves: &std::collections::HashSet<usize>,
) -> Result<lower::Lowered, crate::ShaderError> {
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
            rsrc2: 5 << 1 | lds_blocks << 15,
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure_with(&p, waves).expect("structure");
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

fn host_shader(lowered: lower::Lowered) -> HostShader {
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
        let shader = host_shader(lower_words(&program(), host_wave_max, 0).expect("lower"));
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
fn split_mask_and_scc_as_data_covers_both_halves() {
    let mut words = vec![sop1(0x03, 5, 0x80)];
    words.extend(s_load_dwordx4(8, 4));
    words.push(vop2(0x1A, 1, int(2), 0));
    // Force console lane numbering and split-wave lowering.
    words.extend(vop3(0x365, 3, 193, int(0), 0));
    for (index, (cmp, bound)) in [(0xC4, 16), (0xC1, 31), (0xC4, 0)].into_iter().enumerate() {
        words.push(vopc(cmp, int(bound), 0));
        words.push(0x8794_6A7E); // s_and_b64 s[20:21], exec, vcc
        words.push(0x850C_8081); // s_cselect_b32 s12, 1, 0
        words.push(vop1(0x01, 2, 12));
        words.extend(buffer_store_dword_offen(2, 1, 8, 256 * index as u16));
    }
    words.push(S_ENDPGM);
    let shader = host_shader(lower_words(&words, 32, 0).expect("split AND SCC lowers"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(out) = run_hardware(&shader, 64 * 3) {
        for lane in 0..64 {
            assert_eq!(out[lane], 1, "low-half bit, lane {lane}");
            assert_eq!(out[64 + lane], 1, "high-half bit, lane {lane}");
            assert_eq!(out[128 + lane], 0, "empty mask, lane {lane}");
        }
    }

    // An OR can retain bits outside EXEC; do not infer its SCC this way.
    let mut unsupported = words.clone();
    let and = unsupported.iter().position(|&w| w == 0x8794_6A7E).unwrap();
    unsupported[and] = 0x8894_6A7E; // s_or_b64 s[20:21], exec, vcc
    assert!(matches!(
        lower_words(&unsupported, 32, 0),
        Err(crate::ShaderError::Unsupported { what: "a lane mask read as data", .. })
    ));
}

#[test]
fn split_mask_and_scc_in_a_divergent_block_is_refused() {
    let mut words = vec![sop1(0x03, 5, 0x80)];
    words.extend(s_load_dwordx4(8, 4));
    words.push(vop2(0x1A, 1, int(2), 0));
    words.extend(vop3(0x365, 3, 193, int(0), 0));
    words.push(vopc(0xC4, int(40), 0));
    words.push(sop1(0x24, 22, VCC)); // s_and_saveexec_b64 s[22:23], vcc
    words.push(0xBF88_0006); // s_cbranch_execz over the block
    words.push(vopc(0xC1, int(31), 0));
    words.push(0x8794_6A7E); // s_and_b64 s[20:21], exec, vcc
    words.push(0x850C_8081); // s_cselect_b32 s12, 1, 0
    words.push(vop1(0x01, 2, 12));
    words.extend(buffer_store_dword_offen(2, 1, 8, 0));
    words.push(S_ENDPGM);
    assert!(matches!(
        lower_words(&words, 32, 0),
        Err(crate::ShaderError::Unsupported { what: "a lane mask read as data", .. })
    ));
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
    assert!(lower_words(&w, 64, 0).is_ok());
    let lowered = lower_words(&w, 32, 0).expect("split lowers it");
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

#[test]
fn split_waves_refuse_unfenced_lds_writes_in_a_divergent_region() {
    let mut words = vec![
        vop2(0x1A, 1, int(2), 0),
        vopc(0xC4, int(40), 0),
        sop1(0x24, 20, VCC),
        0xBF88_0000 | 4, // s_cbranch_execz over both LDS writes
        0xD800_0000 | 0x0D << 18, // ds_write_b32 v1, v0
        1,
        0xD800_0000 | 0x0D << 18 | 4, // ds_write_b32 v1, v0 offset:4
        1,
        S_ENDPGM,
    ];
    assert!(lower_words(&words, 64, 1).is_ok());
    assert!(matches!(
        lower_words(&words, 32, 1),
        Err(crate::ShaderError::Unsupported { what: "LDS accesses without a uniform LDS fence in split wave64", .. })
    ));
    words[3] = 0xBF88_0000 | 2;
    words.drain(6..8);
    assert!(lower_words(&words, 32, 1).is_ok(), "a lone write doesn't need an in-region fence");
}

#[test]
fn split_waves_fence_lds_once_an_execz_skip_falls_through() {
    let mut words = vec![
        vop2(0x1A, 1, int(2), 0),
        vopc(0xC4, int(40), 0),
        sop1(0x24, 20, VCC),
        0xBF88_0004, // s_cbranch_execz over both LDS writes
        0xD800_0000 | 0x0D << 18, // ds_write_b32 v1, v0
        1,
        0xD800_0000 | 0x0D << 18 | 4, // ds_write_b32 v1, v0 offset:4
        1,
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, true, 256).unwrap();
    assert_eq!(super::split::execz_skip_over(&p, 24), Some(12));
    assert_eq!(super::split::execz_skip_over(&p, 32), None, "the skip's target is past it");
    assert!(super::split::skips_forward(&p, 12));
    words[3] = 0xBF80_0000; // s_nop 0: EXEC alone masks the writes.
    assert!(lower_words(&words, 32, 1).is_ok(), "guarded writes in a convergent block are fenced");
}

#[test]
fn split_waves_run_a_lane_loop_latched_on_the_waves_scc() {
    let mut words = vec![sop1(0x03, 5, 0x80)];
    words.extend(s_load_dwordx4(8, 4));
    words.extend([
        vop2(0x1A, 1, int(2), 0), // Per-lane byte offset.
        sop1(0x04, 30, 126),      // s_mov_b64 s[30:31], exec
        vop1(0x01, 4, int(0)),
        0xD800_0000 | 0x0D << 18, // loop: ds_write_b32 v1, v4
        4 << 8 | 1,
        0xD800_0000 | 0x0D << 18 | 4, // ds_write_b32 v1, v4 offset:4
        4 << 8 | 1,
        vop2(0x25, 4, int(1), 4), // Lane v0 runs v0 + 1 iterations.
        vopc(0xC3, vgpr(4), 0),   // v_cmp_le_u32 vcc, v4, v0
        0x8000_0000 | 0x0F << 23 | 126 << 16 | 106 << 8 | 126, // s_and_b64 exec, exec, vcc
        0xBF85_FFF8,              // s_cbranch_scc1 loop: while any lane is left
        sop1(0x04, 126, 30),
    ]);
    words.extend(buffer_store_dword_offen(4, 1, 8, 0));
    words.push(S_ENDPGM);
    assert!(matches!(
        lower_words(&words, 32, 1),
        Err(crate::ShaderError::Unsupported { what: super::split::UNFENCED_LDS, .. })
    ));
    let waves = std::collections::HashSet::from([52]);
    let shader = host_shader(lower_words_with(&words, 32, 1, &waves).expect("a wave-level latch is fenced"));
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(out) = run_hardware(&shader, 64) {
        assert_eq!(out, (0..64).map(|lane| lane + 1).collect::<Vec<_>>());
    }
}

/// A lane loop left through `exec_exit` (`s_cbranch_execz` or `vccz`, at
/// byte 56) and latched by `s_branch`: each lane v0 runs v0 + 1 iterations.
fn loop_with_exit(vcc: bool) -> Vec<u32> {
    let mut words = vec![sop1(0x03, 5, 0x80)];
    words.extend(s_load_dwordx4(8, 4));
    words.extend([
        vop2(0x1A, 1, int(2), 0),
        sop1(0x04, 30, 126), // s_mov_b64 s[30:31], exec
        vop1(0x01, 4, int(0)),
        0xD800_0000 | 0x0D << 18, // loop: ds_write_b32 v1, v4
        4 << 8 | 1,
        0xD800_0000 | 0x0D << 18 | 4, // ds_write_b32 v1, v4 offset:4
        4 << 8 | 1,
        vop2(0x25, 4, int(1), 4),
        vopc(0xC3, vgpr(4), 0), // v_cmp_le_u32 vcc, v4, v0
    ]);
    if vcc {
        words.extend([
            0x8000_0000 | 0x0F << 23 | 106 << 16 | 106 << 8 | 126, // s_and_b64 vcc, exec, vcc
            sop1(0x04, 126, 106),                                 // s_mov_b64 exec, vcc
            0xBF86_0001,                                          // s_cbranch_vccz exit
            0xBF82_FFF6,                                          // s_branch loop
        ]);
    } else {
        words.extend([
            0x8000_0000 | 0x0F << 23 | 126 << 16 | 106 << 8 | 126, // s_and_b64 exec, exec, vcc
            vop1(0x01, 5, int(0)),                                // v_mov_b32 v5, 0 (pad to byte 56)
            0xBF88_0001,                                          // s_cbranch_execz exit
            0xBF82_FFF6,                                          // s_branch loop
        ]);
    }
    words.push(sop1(0x04, 126, 30)); // exit: s_mov_b64 exec, s[30:31]
    words.extend(buffer_store_dword_offen(4, 1, 8, 0));
    words.push(S_ENDPGM);
    words
}

#[test]
fn split_waves_run_lane_loops_left_on_the_waves_exec_or_vcc() {
    for vcc in [false, true] {
        let words = loop_with_exit(vcc);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ir::build(&code, true, 256).unwrap();
        assert!(!super::split::skips_forward(&p, 56), "a loop's exit does not fall through");
        assert!(matches!(
            lower_words(&words, 32, 1),
            Err(crate::ShaderError::Unsupported { what: super::split::UNFENCED_LDS, .. })
        ));
        let waves = std::collections::HashSet::from([56]);
        let shader = host_shader(lower_words_with(&words, 32, 1, &waves).expect("a wave-level exit is fenced"));
        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(out) = run_hardware(&shader, 64) {
            assert_eq!(out, (0..64).map(|lane| lane + 1).collect::<Vec<_>>(), "vcc: {vcc}");
        }
    }
}

#[test]
fn split_waves_fence_lds_in_a_loop_only_scalars_decide() {
    let mut words = vec![
        vop2(0x1A, 1, int(2), 0), // Per-lane byte offset.
        sop1(0x03, 6, 0x80),      // s_mov_b32 s6, 0
        0xD800_0000 | 0x0D << 18, // loop: ds_write_b32 v1, v0
        1,
        0xD800_0000 | 0x0D << 18 | 4, // ds_write_b32 v1, v0 offset:4
        1,
        0x8000_0000 | 6 << 16 | 0x81 << 8 | 6, // s_add_u32 s6, s6, 1
        0xBF00_0000 | 0x0A << 16 | 0x82 << 8 | 6, // s_cmp_lt_u32 s6, 2
        0xBF85_FFF9, // s_cbranch_scc1 loop
        S_ENDPGM,
    ];
    assert!(lower_words(&words, 32, 1).is_ok(), "both halves take the same iterations");
    words[1] = vop1(0x02, 6, vgpr(0)); // v_readfirstlane_b32 s6, v0: each half's own lane
    assert!(matches!(
        lower_words(&words, 32, 1),
        Err(crate::ShaderError::Unsupported { what: "LDS accesses without a uniform LDS fence in split wave64", .. })
    ));
}

#[test]
fn split_waves_rejoin_after_a_loop_with_a_separate_latch() {
    let mut words = vec![sop1(0x03, 5, 0x80)];
    words.extend(s_load_dwordx4(8, 4));
    words.extend([
        vop2(0x1A, 1, int(2), 0), // Per-lane byte offset.
        vop1(0x01, 4, int(0)),
        0xBF82_0001, // Enter at the loop test, not the latch.
        vop2(0x25, 4, int(1), 4), // Latch: increase the counter.
        vopc(0xC4, int(2), 4),
        0xBF87_FFFD, // While counter < 2, go back via the latch.
        vop2(0x25, 4, vgpr(0), 4),
        0xD800_0000 | 0x0D << 18, // ds_write_b32 v1, v4
        4 << 8 | 1,
        vop2(0x1D, 3, u16::from(LITERAL), 1),
        128, // Same lane of the other host wave.
        0xD800_0000 | 0x36 << 18, // ds_read_b32 v2, v3
        2 << 24 | 3,
    ]);
    words.extend(buffer_store_dword_offen(2, 1, 8, 0));
    words.push(S_ENDPGM);
    let shader = host_shader(lower_words(&words, 32, 1).expect("lower loop and LDS exchange"));
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(out) = run_hardware(&shader, 64) {
        assert_eq!(out, (0..64).map(|lane| (lane ^ 32) + 2).collect::<Vec<_>>());
    }
}

#[test]
fn split_waves_allow_only_disjoint_lds_stores_in_a_lane_prefix() {
    let mut words = vec![sop1(0x03, 5, 0x80)];
    words.extend(s_load_dwordx4(8, 4));
    words.extend(vop3(0x365, 2, 193, int(0), 0));
    words.extend(vop3(0x366, 2, 193, vgpr(2), 0));
    words.extend([
        vop2(0x25, 4, int(1), 2),
        vopc(0xC4, int(40), 2),
        sop1(0x24, 20, VCC),
        0xBF88_0006, // Only lanes 0–39 store; all lanes read after the join.
        vop2(0x1A, 7, int(1), 2),
        vop2(0x1A, 10, int(3), 2),
        0xD800_0000 | 0x1F << 18 | 320, // Halfwords after the dword pairs.
        4 << 8 | 7,
        0xD800_0000 | 0x0E << 18 | 1 << 8,
        4 << 16 | 4 << 8 | 10,
        0x8000_0000 | 0x11 << 23 | 126 << 16 | 20 << 8 | 126, // Restore EXEC.
        vop2(0x1D, 3, int(32), 2),
        vop2(0x13, 3, int(39), 3),
        vop2(0x1A, 7, int(1), 3),
        vop2(0x1A, 10, int(3), 3),
        0xD800_0000 | 0x3C << 18 | 320, // ds_read_u16 v5, v7
        5 << 24 | 7,
        0xD800_0000 | 0x36 << 18 | 4, // ds_read_b32 v6, v10 offset:4
        6 << 24 | 10,
        vop2(0x25, 5, vgpr(6), 5),
        vop2(0x1A, 1, int(2), 2),
    ]);
    words.extend(buffer_store_dword_offen(5, 1, 8, 0));
    words.push(S_ENDPGM);
    let shader = host_shader(lower_words(&words, 32, 1).expect("lower disjoint stores"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(out) = run_hardware(&shader, 64) {
        assert_eq!(out, (0..64).map(|lane| 2 * ((lane ^ 32_u32).min(39) + 1)).collect::<Vec<_>>());
    }
    words[13] = 0xD800_0000 | 0x1F << 18 | 318;
    assert!(matches!(
        lower_words(&words, 32, 1),
        Err(crate::ShaderError::Unsupported {
            what: "LDS accesses without a uniform LDS fence in split wave64", ..
        })
    ));
}

#[test]
fn split_waves_allow_disjoint_fixed_lds_dwords_in_one_arm() {
    let mut words = vec![
        vopc(0xD2, int(0), 0), // Only thread x == 0.
        0xBF88_0005,
        vop1(0x01, 2, int(0)),
        0xD800_0000 | 0x36 << 18 | 4, // Read dword 1.
        3 << 24 | 2,
        0xD800_0000 | 0x0D << 18, // Write dword 0.
        3 << 8 | 2,
        S_ENDPGM,
    ];
    assert!(lower_words(&words, 32, 1).is_ok());
    words[0] = vopc(0xD4, int(40), 0);
    assert!(matches!(
        lower_words(&words, 32, 1),
        Err(crate::ShaderError::Unsupported {
            what: "LDS accesses without a uniform LDS fence in split wave64", ..
        })
    ), "fixed writes by lanes in both halves are not proved independent");
    words[0] = vopc(0xD2, int(0), 0);
    words[3] &= !4;
    assert!(lower_words(&words, 32, 1).is_ok(), "thread zero alone cannot race itself");
    words[3] |= 4;
    words[1] = 0xBF88_0007;
    words.splice(7..7, [0xD800_0000 | 0x36 << 18 | 8, 4 << 24 | 2]);
    assert!(lower_words(&words, 32, 1).is_ok(), "thread zero alone cannot race itself");
    words.drain(7..9);
    words[1] = 0xBF88_0005;
    words[2] = vop1(0x01, 2, vgpr(0));
    assert!(lower_words(&words, 32, 1).is_ok(), "thread zero alone cannot race itself");
    words[2] = vop1(0x01, 2, int(0));
    words[5] = 0xD800_0000 | 0x4D << 18; // Eight bytes overlap the read.
    assert!(lower_words(&words, 32, 1).is_ok(), "thread zero alone cannot race itself");
    words[5] = 0xD800_0000 | 0x20 << 18; // Atomics are not plain stores.
    assert!(lower_words(&words, 32, 1).is_ok(), "thread zero alone cannot race itself");

    let mut stores = vec![
        vopc(0xD2, int(0), 0),
        0xBF88_0005,
        vop1(0x01, 2, int(0)),
        0xD800_0000 | 0x0E << 18 | 2 | 3 << 8,
        2 << 16 | 2 << 8 | 2,
        0xD800_0000 | 0x0E << 18 | 1 << 8,
        2 << 16 | 2 << 8 | 2,
        S_ENDPGM,
    ];
    assert!(lower_words(&stores, 32, 1).is_ok());
    stores[5] |= 2;
    assert!(lower_words(&stores, 32, 1).is_ok(), "thread zero alone cannot race itself");
}

#[test]
fn split_waves_recognize_a_saved_singleton_lds_predicate() {
    let mut words = vec![
        0xD4C2_0000, 0x0002_0080, // v_cmp_eq_u32 s[0:1], 0, v0
        0xBF82_0001, // Skip a block before using the saved predicate.
        vop1(0x01, 3, int(0)),
        0xBE84_2400, // s_and_saveexec_b64 s[4:5], s[0:1]
        0xBF88_0005,
        vop1(0x01, 2, int(0)),
        0xD800_0000 | 0x36 << 18 | 4,
        3 << 24 | 2,
        0xD800_0000 | 0x0D << 18,
        3 << 8 | 2,
        S_ENDPGM,
    ];
    assert!(lower_words(&words, 32, 1).is_ok());
    words[1] = 0x0002_0081; // Compare thread x with 1 instead of zero.
    assert!(matches!(
        lower_words(&words, 32, 1),
        Err(crate::ShaderError::Unsupported {
            what: "LDS accesses without a uniform LDS fence in split wave64", ..
        })
    ));
}

#[test]
fn split_waves_order_lds_accesses_for_a_derived_singleton() {
    let mut words = vec![sop1(0x03, 5, 0x80)];
    words.extend(s_load_dwordx4(8, 4));
    words.extend([
        sop1(0x04, 20, 126), // Save EXEC.
        vop1(0x01, 2, int(0)),
        vopc(0xD2, int(0), 0), // Only thread zero sets the flag.
    ]);
    let first_branch = words.len();
    words.push(0);
    words.extend([
        vop1(0x01, 7, int(0)),
        vop1(0x01, 4, int(5)),
        0xD800_0000 | 0x0D << 18, 4 << 8 | 7,
        vop1(0x01, 2, 193),
    ]);
    words[first_branch] = 0xBF88_0000 | (words.len() - first_branch - 1) as u32;
    words.extend([
        sop1(0x04, 126, 20),
        vopc(0xD2, 193, 2),
    ]);
    let second_branch = words.len();
    words.push(0);
    words.extend([
        vop1(0x01, 7, int(0)),
        0xD800_0000 | 0x36 << 18, 3 << 24 | 7,
        vop1(0x01, 4, int(9)),
        0xD800_0000 | 0x0D << 18, 4 << 8 | 7,
        0xD800_0000 | 0x0D << 18 | 4, 4 << 8 | 7,
    ]);
    words[second_branch] = 0xBF88_0000 | (words.len() - second_branch - 1) as u32;
    words.extend([
        sop1(0x04, 126, 20),
        vop1(0x01, 7, int(0)),
        0xD800_0000 | 0x36 << 18, 3 << 24 | 7,
        vop2(0x1A, 1, int(2), 0),
    ]);
    words.extend(buffer_store_dword_offen(3, 1, 8, 0));
    words.push(S_ENDPGM);
    let shader = host_shader(lower_words(&words, 32, 1).expect("lower singleton LDS accesses"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(out) = run_hardware(&shader, 64) {
        assert_eq!(out, vec![9; 64]);
    }
    words[4] = vop1(0x01, 2, 193); // The flag no longer excludes other threads.
    assert!(matches!(
        lower_words(&words, 32, 1),
        Err(crate::ShaderError::Unsupported {
            what: "LDS accesses without a uniform LDS fence in split wave64", ..
        })
    ));
    words[4] = vop1(0x01, 2, int(0));
    words[5] = vopc(0xD4, int(40), 0); // Multiple threads set the flag.
    assert!(matches!(
        lower_words(&words, 32, 1),
        Err(crate::ShaderError::Unsupported {
            what: "LDS accesses without a uniform LDS fence in split wave64", ..
        })
    ));
}
