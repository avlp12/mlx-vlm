// The floor question, with nothing in the way: how long is one
// GPU-signal -> host-observe -> host-set -> GPU-resume round trip?
import Foundation
import Metal
import QuartzCore

let N = Int(ProcessInfo.processInfo.environment["FENCE_N"] ?? "101")!
let REPS = Int(ProcessInfo.processInfo.environment["FENCE_REPS"] ?? "20")!
let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
func median(_ v: [Double]) -> Double { let s = v.sorted(); return s[s.count/2] }

// 1) how expensive is reading MTLSharedEvent.signaledValue? (polling granularity)
let ev0 = dev.makeSharedEvent()!
var sink: UInt64 = 0
let tR0 = CACurrentMediaTime()
for _ in 0..<200_000 { sink &+= ev0.signaledValue }
let readNs = (CACurrentMediaTime() - tR0) * 1e9 / 200_000.0

// 2) pure fence ping-pong: NO GPU work between fences at all
var pure: [Double] = []
for _ in 0..<REPS {
    let ev = dev.makeSharedEvent()!
    let done = DispatchSemaphore(value: 0)
    let th = Thread {
        for i in 0..<N {
            let want = UInt64(2*i + 1)
            while ev.signaledValue < want {}
            ev.signaledValue = want + 1
        }
        done.signal()
    }
    th.qualityOfService = .userInteractive
    th.start()
    let t0 = CACurrentMediaTime()
    let cb = queue.makeCommandBuffer()!
    for i in 0..<N {
        cb.encodeSignalEvent(ev, value: UInt64(2*i + 1))
        cb.encodeWaitForEvent(ev, value: UInt64(2*i + 2))
    }
    cb.commit(); cb.waitUntilCompleted()
    pure.append(CACurrentMediaTime() - t0)
    done.wait()
}

// 3) one-way GPU->host: single signal, host spins, compare to cb.gpuEndTime
var oneway: [Double] = []
for _ in 0..<REPS {
    let ev = dev.makeSharedEvent()!
    var obs = 0.0
    let done = DispatchSemaphore(value: 0)
    let th = Thread { while ev.signaledValue < 1 {}; obs = CACurrentMediaTime(); done.signal() }
    th.qualityOfService = .userInteractive
    th.start()
    let cb = queue.makeCommandBuffer()!
    cb.encodeSignalEvent(ev, value: 1)
    cb.commit(); cb.waitUntilCompleted()
    done.wait()
    oneway.append(obs - cb.gpuEndTime)
}

// 4) empty command buffer submit+complete: the other half of today's model
var empty: [Double] = []
for _ in 0..<REPS {
    let t0 = CACurrentMediaTime()
    for _ in 0..<N { let cb = queue.makeCommandBuffer()!; cb.commit(); cb.waitUntilCompleted() }
    empty.append(CACurrentMediaTime() - t0)
}

func us(_ x: Double) -> String { String(format: "%.1f", x * 1e6 / Double(N)) }
print("""
{
 "device": "\(dev.name)", "N": \(N), "reps": \(REPS),
 "signaledValue_read_ns": \(String(format: "%.1f", readNs)),
 "pure_fence_roundtrip_us": \(us(median(pure))),
 "oneway_gpu_signal_to_host_observe_us": \(String(format: "%.1f", median(oneway) * 1e6)),
 "empty_cmdbuf_submit_complete_us": \(us(median(empty)))
}
""")
