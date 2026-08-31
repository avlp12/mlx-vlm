// Three-component decomposition of an evented GPU<->host crossing, as required
// before any GO: (a) encoder close/reopen, (b) MTLSharedEvent round trip,
// (c) the GPU pipeline drain/refill a mid-buffer wait forces.
// Min over reps: contention can only inflate a timing.
import Foundation
import Metal
import QuartzCore

let N = Int(ProcessInfo.processInfo.environment["FENCE_N"] ?? "101")!
let REPS = Int(ProcessInfo.processInfo.environment["FENCE_REPS"] ?? "60")!
let NELEM = Int(ProcessInfo.processInfo.environment["FENCE_NELEM"] ?? "262144")!
let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void bump(device float* a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float v = a[i];
  for (int k = 0; k < 8; ++k) { v = fma(v, 1.0000001f, 1.0f); }
  a[i] = v;
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "bump")!)
let buf = dev.makeBuffer(length: NELEM * 4, options: .storageModeShared)!
let tg = MTLSize(width: pso.threadExecutionWidth, height: 1, depth: 1)
let grid = MTLSize(width: NELEM, height: 1, depth: 1)
func dispatchInto(_ e: MTLComputeCommandEncoder) {
    e.setComputePipelineState(pso); e.setBuffer(buf, offset: 0, index: 0)
    e.dispatchThreads(grid, threadsPerThreadgroup: tg)
}
func bench(_ f: () -> Void) -> Double {
    for _ in 0..<5 { f() }
    var t: [Double] = []
    for _ in 0..<REPS { let a = CACurrentMediaTime(); f(); t.append(CACurrentMediaTime() - a) }
    return t.min()!
}
func us(_ x: Double) -> String { String(format: "%.1f", x * 1e6 / Double(N)) }

// E0: all N dispatches inside ONE encoder
let e0 = bench {
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    for _ in 0..<N { dispatchInto(e) }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
}
// E1: one encoder per dispatch (close/reopen), no events
let e1 = bench {
    let cb = queue.makeCommandBuffer()!
    for _ in 0..<N { let e = cb.makeComputeCommandEncoder()!; dispatchInto(e); e.endEncoding() }
    cb.commit(); cb.waitUntilCompleted()
}
// E2: pure MTLSharedEvent ping-pong, NO encoders, NO dispatches
let e2 = bench {
    let ev = dev.makeSharedEvent()!
    let done = DispatchSemaphore(value: 0)
    let th = Thread { for i in 0..<N { let w = UInt64(2*i+1); while ev.signaledValue < w {}; ev.signaledValue = w+1 }; done.signal() }
    th.qualityOfService = .userInteractive; th.start()
    let cb = queue.makeCommandBuffer()!
    for i in 0..<N { cb.encodeSignalEvent(ev, value: UInt64(2*i+1)); cb.encodeWaitForEvent(ev, value: UInt64(2*i+2)) }
    cb.commit(); cb.waitUntilCompleted(); done.wait()
}
// E3: the real thing -- dispatch, close, signal, wait, live host
let e3 = bench {
    let ev = dev.makeSharedEvent()!
    let done = DispatchSemaphore(value: 0)
    let th = Thread { for i in 0..<N { let w = UInt64(2*i+1); while ev.signaledValue < w {}; ev.signaledValue = w+1 }; done.signal() }
    th.qualityOfService = .userInteractive; th.start()
    let cb = queue.makeCommandBuffer()!
    for i in 0..<N {
        let e = cb.makeComputeCommandEncoder()!; dispatchInto(e); e.endEncoding()
        cb.encodeSignalEvent(ev, value: UInt64(2*i+1)); cb.encodeWaitForEvent(ev, value: UInt64(2*i+2))
    }
    cb.commit(); cb.waitUntilCompleted(); done.wait()
}
print("""
{
 "device": "\(dev.name)", "N": \(N), "nelem": \(NELEM), "reps": \(REPS), "stat": "min",
 "E0_one_encoder_us":        \(us(e0)),
 "E1_encoder_per_op_us":     \(us(e1)),
 "E2_pure_event_pingpong_us":\(us(e2)),
 "E3_full_evented_us":       \(us(e3)),
 "component_a_encoder_close_reopen_us": \(us(e1 - e0)),
 "component_b_event_roundtrip_us":      \(us(e2)),
 "component_c_drain_refill_us":         \(us(e3 - e1 - e2))
}
""")
