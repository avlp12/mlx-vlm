// P1 gate: what does a GPU-timeline fence round-trip actually cost on M3 Ultra?
//
// The proposed MLX patch would, inside one command buffer, encode
//   ... GPU work ... signal(e, 2i+1) ; wait(e, 2i+2) ... GPU work ...
// and have a persistent host thread notice 2i+1, run a ~20us jaccl all_reduce
// on the unified-memory buffer, then set 2i+2.  This measures the floor of
// that mechanism, with no MLX and no networking involved.
//
// V0  N dispatches, one command buffer, no fences          -> baseline
// V1  N dispatches, N command buffers, CPU commit+wait each -> today's model
// V2  N dispatches, one command buffer, fence pairs, host SPINS
// V3  N dispatches, one command buffer, fence pairs, host woken by listener

import Foundation
import Metal
import QuartzCore

let N = Int(ProcessInfo.processInfo.environment["FENCE_N"] ?? "101")!
let REPS = Int(ProcessInfo.processInfo.environment["FENCE_REPS"] ?? "20")!
let HOST_US = Double(ProcessInfo.processInfo.environment["FENCE_HOST_US"] ?? "20")!
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

func busyWaitUs(_ us: Double) {
    if us <= 0 { return }
    let t0 = DispatchTime.now().uptimeNanoseconds
    let n = UInt64(us * 1000.0)
    while DispatchTime.now().uptimeNanoseconds - t0 < n {}
}

func encodeDispatch(_ cb: MTLCommandBuffer) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(buf, offset: 0, index: 0)
    enc.dispatchThreads(MTLSize(width: NELEM, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: pso.threadExecutionWidth, height: 1, depth: 1))
    enc.endEncoding()
}

func median(_ v: [Double]) -> Double { let s = v.sorted(); return s[s.count/2] }

// ---------------------------------------------------------------- V0
func v0() -> (Double, Double) {
    var wall: [Double] = [], gpu: [Double] = []
    for _ in 0..<REPS {
        let t0 = CACurrentMediaTime()
        let cb = queue.makeCommandBuffer()!
        for _ in 0..<N { encodeDispatch(cb) }
        cb.commit(); cb.waitUntilCompleted()
        wall.append(CACurrentMediaTime() - t0)
        gpu.append(cb.gpuEndTime - cb.gpuStartTime)
    }
    return (median(wall), median(gpu))
}

// ---------------------------------------------------------------- V1
func v1() -> (Double, Double) {
    var wall: [Double] = [], gpu: [Double] = []
    for _ in 0..<REPS {
        let t0 = CACurrentMediaTime()
        var g = 0.0
        for _ in 0..<N {
            let cb = queue.makeCommandBuffer()!
            encodeDispatch(cb)
            cb.commit(); cb.waitUntilCompleted()
            g += cb.gpuEndTime - cb.gpuStartTime
            busyWaitUs(HOST_US)
        }
        wall.append(CACurrentMediaTime() - t0); gpu.append(g)
    }
    return (median(wall), median(gpu))
}

// ---------------------------------------------------------------- V2 (spin)
func v2() -> (Double, Double) {
    var wall: [Double] = [], gpu: [Double] = []
    for _ in 0..<REPS {
        let ev = dev.makeSharedEvent()!
        let done = DispatchSemaphore(value: 0)
        let th = Thread {
            for i in 0..<N {
                let want = UInt64(2*i + 1)
                while ev.signaledValue < want { /* spin: pre-spun, polling */ }
                busyWaitUs(HOST_US)
                ev.signaledValue = want + 1
            }
            done.signal()
        }
        th.qualityOfService = .userInteractive
        th.start()

        let t0 = CACurrentMediaTime()
        let cb = queue.makeCommandBuffer()!
        for i in 0..<N {
            encodeDispatch(cb)
            cb.encodeSignalEvent(ev, value: UInt64(2*i + 1))
            cb.encodeWaitForEvent(ev, value: UInt64(2*i + 2))
        }
        cb.commit(); cb.waitUntilCompleted()
        wall.append(CACurrentMediaTime() - t0)
        gpu.append(cb.gpuEndTime - cb.gpuStartTime)
        done.wait()
    }
    return (median(wall), median(gpu))
}

// ---------------------------------------------------------------- V3 (woken)
func v3() -> (Double, Double) {
    var wall: [Double] = [], gpu: [Double] = []
    let q = DispatchQueue(label: "fence.listener", qos: .userInteractive)
    for _ in 0..<REPS {
        let ev = dev.makeSharedEvent()!
        let listener = MTLSharedEventListener(dispatchQueue: q)
        for i in 0..<N {
            ev.notify(listener, atValue: UInt64(2*i + 1)) { e, _ in
                busyWaitUs(HOST_US)
                e.signaledValue = UInt64(2*i + 2)
            }
        }
        let t0 = CACurrentMediaTime()
        let cb = queue.makeCommandBuffer()!
        for i in 0..<N {
            encodeDispatch(cb)
            cb.encodeSignalEvent(ev, value: UInt64(2*i + 1))
            cb.encodeWaitForEvent(ev, value: UInt64(2*i + 2))
        }
        cb.commit(); cb.waitUntilCompleted()
        wall.append(CACurrentMediaTime() - t0)
        gpu.append(cb.gpuEndTime - cb.gpuStartTime)
    }
    return (median(wall), median(gpu))
}

// warm
_ = v0()
let (w0, g0) = v0()
let (w1, g1) = v1()
let (w2, g2) = v2()
let (w3, g3) = v3()

func us(_ x: Double) -> String { String(format: "%.1f", x * 1e6 / Double(N)) }
func ms(_ x: Double) -> String { String(format: "%.3f", x * 1e3) }

print("""
{
 "device": "\(dev.name)", "N": \(N), "reps": \(REPS), "host_us": \(HOST_US), "nelem": \(NELEM),
 "V0_nofence":        {"wall_ms": \(ms(w0)), "gpu_ms": \(ms(g0)), "per_op_us": \(us(w0))},
 "V1_cmdbuf_per_op":  {"wall_ms": \(ms(w1)), "gpu_ms": \(ms(g1)), "per_op_us": \(us(w1))},
 "V2_fence_spin":     {"wall_ms": \(ms(w2)), "gpu_ms": \(ms(g2)), "per_op_us": \(us(w2))},
 "V3_fence_listener": {"wall_ms": \(ms(w3)), "gpu_ms": \(ms(g3)), "per_op_us": \(us(w3))},
 "added_per_fence_us": {
   "V1_minus_V0": \(us(w1 - w0)),
   "V2_minus_V0": \(us(w2 - w0)),
   "V3_minus_V0": \(us(w3 - w0))
 }
}
""")
