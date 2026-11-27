use super::capture;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

const WRITER: &str = r#"
#define RS "UAV(u0), RootConstants(num32BitConstants=1, b0)"
RWByteAddressBuffer data : register(u0);
cbuffer c : register(b0) { uint increment; }
[RootSignature(RS)]
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint at = id.x * 4;
    data.Store(at, data.Load(at) + increment);
}
"#;

fn writer() -> (ID3D12RootSignature, ID3D12PipelineState) {
    let host = crate::host::host().unwrap();
    let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
    // SAFETY: source and entry point are live for the call.
    let result = unsafe {
        D3DCompile(WRITER.as_ptr().cast(), WRITER.len(), s!("snapshot test"), None, None,
            s!("main"), s!("cs_5_1"), 0, 0, &mut code, Some(&mut errors))
    };
    if let Some(errors) = errors {
        // SAFETY: the error blob owns these bytes.
        let bytes = unsafe { std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize()) };
        assert!(result.is_ok(), "{}", String::from_utf8_lossy(bytes));
    }
    result.unwrap();
    let code = code.unwrap();
    // SAFETY: the shader blob stays alive while the root signature and PSO are created.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
            ..Default::default()
        };
        let pso = host.device.CreateComputePipelineState(&desc).unwrap();
        (root, pso)
    }
}

#[test]
fn sequential_dispatches_capture_their_own_offset_bytes() {
    let Some(host) = crate::host::host() else {
        return; // no host GPU here
    };
    let mut title = [0u32; 16];
    let address = title.as_mut_ptr() as u64;
    crate::shadow::register(address as usize, address, 64);
    let shadow = crate::shadow::find(address, 64).expect("a source shadow");
    for i in 0..16u32 {
        shadow.write_host(address + u64::from(i) * 4, &i.to_ne_bytes());
    }
    let (root, pso) = writer();
    let list = host.new_list().unwrap();
    let dispatch = |increment: u32| {
        // SAFETY: the list is open, and the source and pipeline outlive its submission.
        unsafe {
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list.SetComputeRootUnorderedAccessView(0, shadow.gpu(address));
            list.list.SetComputeRoot32BitConstants(1, 1, (&increment as *const u32).cast(), 0);
            list.list.Dispatch(1, 1, 1);
        }
    };
    let snapshot = || capture(&list, shadow.clone(), address + 20, 32, "snapshot test".into()).expect("a snapshot");
    // The first dispatch promotes the source to UAV before the first capture.
    dispatch(0);
    let before = snapshot();
    assert!(list.take_rebind(), "capture invalidates compute state");
    dispatch(0x100);
    let first = snapshot();
    assert!(list.take_rebind(), "capture invalidates compute state");
    dispatch(0x1000);
    let second = snapshot();
    assert!(list.take_rebind(), "capture invalidates compute state");
    list.close();
    host.execute(&[&list]);
    // SAFETY: execute waits for the list; each capture owns its mapped output.
    unsafe {
        assert_eq!(before.read(), [5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(first.read(), [0x105, 0x106, 0x107, 0x108, 0x109, 0x10a, 0x10b, 0x10c]);
        assert_eq!(second.read(), [0x1105, 0x1106, 0x1107, 0x1108, 0x1109, 0x110a, 0x110b, 0x110c]);
    }
    assert_eq!(shadow.read_host(address, 16), (0..16).map(|i| i + 0x1100).collect::<Vec<_>>());
    crate::shadow::unregister(address as usize, address);
}
