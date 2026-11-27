//! Diagnostic copies of buffer UAVs at a dispatch boundary.

use std::sync::{Arc, LazyLock};

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host};
use crate::shadow::Shadow;

// shortcut: views over 16 MiB are refused, raise the limit if a traced view needs it.
pub(crate) const MAX_BYTES: u64 = 16 << 20;
const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), UAV(u0), UAV(u1)"
cbuffer Params : register(b0) { uint offset; uint count; }
RWByteAddressBuffer source : register(u0);
RWByteAddressBuffer output : register(u1);
[RootSignature(RS)]
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x < count) output.Store(id.x * 4, source.Load(offset + id.x * 4));
}
"#;

struct Copier {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Copier {}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Sync for Copier {}

fn copier() -> Option<&'static Copier> {
    static COPIER: LazyLock<Option<Copier>> = LazyLock::new(|| match make() {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::warn!(error = %e, "buffer snapshot shader unavailable");
            None
        }
    });
    COPIER.as_ref()
}

fn make() -> windows::core::Result<Copier> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: source, entry point and output pointers live through compilation.
    let result = unsafe {
        D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("snapshot"), None, None,
            s!("main"), s!("cs_5_1"), 0, 0, &mut code, Some(&mut errors))
    };
    if let Err(e) = result {
        if let Some(errors) = errors {
            // SAFETY: these bytes belong to the compiler's blob.
            let text = unsafe { std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize()) };
            tracing::warn!("buffer snapshot shader: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the compiled blob includes its root signature and lives through creation.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
            ..Default::default()
        })?;
        Ok(Copier { root, pso })
    }
}

pub(crate) struct Snapshot {
    _output: ID3D12Resource,
    _source: Arc<Shadow>,
    data: usize,
    words: usize,
}
// SAFETY: the mapping is read only after the GPU has completed the submission.
unsafe impl Send for Snapshot {}
// SAFETY: as above; each copy has a private output.
unsafe impl Sync for Snapshot {}

impl Snapshot {
    /// The caller must ensure the submission has completed and no GPU write is in flight.
    #[cfg(test)]
    pub(crate) unsafe fn read(&self) -> Vec<u32> {
        // SAFETY: the allocation contains `words` dwords and the caller waited for the GPU.
        unsafe { std::slice::from_raw_parts(self.data as *const u32, self.words) }.to_vec()
    }
}

fn barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

/// Records a raw UAV copy without changing the shared source's resource state.
pub(crate) fn capture(list: &HostList, shadow: Arc<Shadow>, at: u64, len: u64, who: String) -> Option<Arc<Snapshot>> {
    let (start, size) = shadow.range();
    let valid = len != 0 && len <= MAX_BYTES && len.is_multiple_of(4) && at.is_multiple_of(4)
        && at >= start && at.checked_add(len).zip(start.checked_add(size)).is_some_and(|(end, limit)| end <= limit);
    if !valid {
        tracing::warn!(target: "victory_gpu::trace", "buffer snapshot {who}: invalid or oversized range {at:#x}+{len}");
        return None;
    }
    let gpu = shadow.gpu(at);
    // SAFETY: the shadow owns a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    if !gpu.is_multiple_of(4)
        || !shadow.offset(at).checked_add(len).is_some_and(|end| end <= width)
    {
        tracing::warn!(target: "victory_gpu::trace", "buffer snapshot {who}: unaligned or out-of-bounds host range");
        return None;
    }
    let Some(c) = copier() else { return None };
    let Some(host) = host() else {
        tracing::warn!(target: "victory_gpu::trace", "buffer snapshot {who}: host unavailable");
        return None;
    };
    let Some((output, data)) = host.shared_buffer(len) else {
        tracing::warn!(target: "victory_gpu::trace", "buffer snapshot {who}: allocation failed");
        return None;
    };
    let constants = [(gpu & 15) as u32, (len / 4) as u32];
    let snapshot = Arc::new(Snapshot { _output: output, _source: shadow, data: data as usize, words: (len / 4) as usize });
    // SAFETY: both root UAVs name live buffers, the dispatch is bounded to the validated range,
    // and UAV barriers order the title's writes and subsequent accesses without a source transition.
    unsafe {
        let l = &list.list;
        l.ResourceBarrier(&[barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu & !15);
        l.SetComputeRootUnorderedAccessView(2, snapshot._output.GetGPUVirtualAddress());
        l.Dispatch(constants[1].div_ceil(256), 1, 1);
        l.ResourceBarrier(&[barrier()]);
    }
    list.need_rebind();
    let kept = snapshot.clone();
    list.defer_after_checked(move |completed| {
        if !completed {
            tracing::warn!(target: "victory_gpu::trace", "buffer snapshot {who}: submission completion unconfirmed");
            // The GPU may still be using both allocations after the list is reset.
            std::mem::forget(kept.clone());
            return;
        }
        // SAFETY: the checked callback confirms the GPU completed this submission.
        let words = unsafe { std::slice::from_raw_parts(kept.data as *const u32, kept.words) };
        let set = words.iter().filter(|&&x| x != 0).count();
        let last = words.last().copied().unwrap_or(0);
        let counter = words.get(511).copied();
        tracing::info!(target: "victory_gpu::trace", "buffer {who} right after dispatch: {set} of {} dwords nonzero, last {last:#010x}, dword 511 {counter:?}", words.len());
    });
    Some(snapshot)
}

pub(crate) fn record(list: &HostList, shadow: Arc<Shadow>, at: u64, len: u64, who: String) -> bool {
    capture(list, shadow, at, len, who).is_some()
}

#[cfg(test)]
#[path = "snapshot_tests.rs"]
mod snapshot_tests;
