using System;
using System.Diagnostics;
using System.IO;
using ZeroPrimitives.Memory;
using ZeroTensor.Core;
using ZeroTensor.Core.Neural;
using ZeroTensor.Core.Storage;

namespace ZeroTensor.Benchmarks
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("================================================================================");
            Console.WriteLine("                ZeroTensor High-Performance Compute Benchmark                   ");
            Console.WriteLine("================================================================================");
            Console.WriteLine($"OS: {Environment.OSVersion}");
            Console.WriteLine($"Processor Count: {Environment.ProcessorCount}");
            Console.WriteLine($".NET Runtime: {Environment.Version}");
            Console.WriteLine($"SIMD Vector<float> Count: {System.Numerics.Vector<float>.Count}");
            Console.WriteLine($"SIMD Hardware Accelerated: {System.Numerics.Vector.IsHardwareAccelerated}");
            Console.WriteLine("================================================================================\n");

            // Warm up
            WarmUp();

            // Run benchmarks
            RunVectorAddBenchmark();
            RunGemmBenchmark();
            RunBatchedMatMulBenchmark();
            RunSoftmaxBenchmark();
            RunNormalizationBenchmark();
            RunAttentionBenchmark();
            RunVisionConv2DBenchmark();
            RunZeroCopySliceBenchmark();
            RunMemoryAllocationBenchmark();

            Console.WriteLine("\nAll benchmarks completed successfully.");
        }

        private static void WarmUp()
        {
            var a = Tensor.Zeros<float>(64, 64);
            var b = Tensor.Ones<float>(64, 64);
            var c = a + b;
            var d = TensorBlas.MatMul(a, b);
        }

        private static void RunVectorAddBenchmark()
        {
            Console.WriteLine("--- 1. Vector Addition Benchmark (N = 1,000,000 FP32 elements) ---");
            int n = 1_000_000;
            var a = Tensor.Ones<float>(n);
            var b = Tensor.Ones<float>(n);

            // Scalar Baseline
            float[] aArr = a.ToArray();
            float[] bArr = b.ToArray();
            float[] cArr = new float[n];

            int iterations = 100;

            // Warmup
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < n; j++) cArr[j] = aArr[j] + bArr[j];
            }

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                for (int j = 0; j < n; j++) cArr[j] = aArr[j] + bArr[j];
            }
            sw.Stop();
            double scalarMs = sw.Elapsed.TotalMilliseconds / iterations;

            // ZeroTensor SIMD Add
            for (int i = 0; i < 5; i++) { var _ = a + b; }
            sw.Restart();
            for (int i = 0; i < iterations; i++)
            {
                var c = a + b;
            }
            sw.Stop();
            double simdMs = sw.Elapsed.TotalMilliseconds / iterations;

            double bytesProcessed = 3.0 * n * sizeof(float); // Read A, Read B, Write C
            double simdGbps = (bytesProcessed / (simdMs / 1000.0)) / (1024 * 1024 * 1024);
            double scalarGbps = (bytesProcessed / (scalarMs / 1000.0)) / (1024 * 1024 * 1024);

            Console.WriteLine($"Scalar C# Loop : {scalarMs:F3} ms | Bandwidth: {scalarGbps:F2} GB/s");
            Console.WriteLine($"ZeroTensor SIMD: {simdMs:F3} ms | Bandwidth: {simdGbps:F2} GB/s | Speedup: {scalarMs / simdMs:F2}x\n");
        }

        private static void RunGemmBenchmark()
        {
            Console.WriteLine("--- 2. GEMM Benchmark (Matrix Multiplication) ---");
            int[] sizes = { 256, 512 };

            foreach (int size in sizes)
            {
                var a = Tensor.Ones<float>(size, size);
                var b = Tensor.Ones<float>(size, size);

                // Warmup
                var _ = TensorBlas.MatMul(a, b);

                int iters = size <= 256 ? 20 : 5;
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < iters; i++)
                {
                    var c = TensorBlas.MatMul(a, b);
                }
                sw.Stop();

                double avgMs = sw.Elapsed.TotalMilliseconds / iters;
                double ops = 2.0 * size * size * size; // 2 * M * N * K FLOPs
                double gflops = (ops / (avgMs / 1000.0)) / 1e9;

                Console.WriteLine($"GEMM FP32 [{size} x {size}]: {avgMs:F2} ms | Compute Throughput: {gflops:F2} GFLOPS");
            }
            Console.WriteLine();
        }

        private static void RunBatchedMatMulBenchmark()
        {
            Console.WriteLine("--- 3. Batched MatMul Benchmark (LLM Multi-Head Attention Projection) ---");
            // [Batch=8, Heads=8, Seq=128, Dim=64] -> flattened batch [64, 128, 64]
            int b = 64, m = 128, k = 64, n = 64;
            var q = Tensor.Ones<float>(b, m, k);
            var kWeight = Tensor.Ones<float>(b, k, n);

            // Warmup
            var _ = TensorBlas.MatMul(q, kWeight);

            int iters = 5;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                var outTensor = TensorBlas.MatMul(q, kWeight);
            }
            sw.Stop();

            double avgMs = sw.Elapsed.TotalMilliseconds / iters;
            double totalFlops = (double)b * 2.0 * m * k * n;
            double gflops = (totalFlops / (avgMs / 1000.0)) / 1e9;

            Console.WriteLine($"Batched MatMul [{b}, {m}, {k}] x [{b}, {k}, {n}]: {avgMs:F2} ms | Throughput: {gflops:F2} GFLOPS\n");
        }

        private static void RunSoftmaxBenchmark()
        {
            Console.WriteLine("--- 4. Softmax Benchmark (N = 4,096 tokens, LLM context row) ---");
            int n = 4096;
            var logits = Tensor.Ones<float>(n);

            // Standard Softmax (via Tensor.Softmax)
            int iters = 200;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                var s = Tensor.Softmax(logits, 0);
            }
            sw.Stop();
            double stdMs = sw.Elapsed.TotalMilliseconds / iters;

            // Fused SoftmaxFast (via TensorOps.SoftmaxFast)
            sw.Restart();
            for (int i = 0; i < iters; i++)
            {
                var s = TensorOps.SoftmaxFast(logits, 0);
            }
            sw.Stop();
            double fastMs = sw.Elapsed.TotalMilliseconds / iters;

            Console.WriteLine($"Softmax (Standard) : {stdMs * 1000:F2} µs");
            Console.WriteLine($"SoftmaxFast (Fused) : {fastMs * 1000:F2} µs\n");
        }

        private static void RunNormalizationBenchmark()
        {
            Console.WriteLine("--- 5. Normalization Benchmark (RMSNorm & LayerNorm, HiddenDim = 4096) ---");
            int hiddenDim = 4096;
            var x = Tensor.Ones<float>(hiddenDim);
            var gamma = Tensor.Ones<float>(hiddenDim);
            var beta = Tensor.Zeros<float>(hiddenDim);

            int iters = 200;

            // RMSNorm
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                var outNorm = TensorOps.RMSNorm(x, gamma);
            }
            sw.Stop();
            double rmsMs = sw.Elapsed.TotalMilliseconds / iters;

            // LayerNorm
            sw.Restart();
            for (int i = 0; i < iters; i++)
            {
                var outNorm = TensorOps.LayerNorm(x, gamma, beta);
            }
            sw.Stop();
            double lnMs = sw.Elapsed.TotalMilliseconds / iters;

            Console.WriteLine($"RMSNorm (LLaMA style) : {rmsMs * 1000:F2} µs");
            Console.WriteLine($"LayerNorm (BERT style) : {lnMs * 1000:F2} µs\n");
        }

        private static void RunAttentionBenchmark()
        {
            Console.WriteLine("--- 6. Attention Benchmark (Scaled Dot-Product Attention) ---");
            int seqLen = 64;
            int headDim = 64;

            var q = Tensor.Ones<float>(seqLen, headDim);
            var k = Tensor.Ones<float>(seqLen, headDim);
            var v = Tensor.Ones<float>(seqLen, headDim);

            // Warmup
            var _ = TensorOps.ScaledDotProductAttention(q, k, v);

            int iters = 20;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                var attn = TensorOps.ScaledDotProductAttention(q, k, v);
            }
            sw.Stop();
            double avgMs = sw.Elapsed.TotalMilliseconds / iters;

            Console.WriteLine($"SDPA [Seq={seqLen}, Dim={headDim}]: {avgMs:F2} ms\n");
        }

        private static void RunVisionConv2DBenchmark()
        {
            Console.WriteLine("--- 7. Computer Vision Conv2D Benchmark (im2col + GEMM) ---");
            // Input: [1, 3, 64, 64], Kernel: [16, 3, 3, 3]
            var input = Tensor.Ones<float>(1, 3, 64, 64);
            var weight = Tensor.Ones<float>(16, 3, 3, 3);

            // Warmup
            var _ = TensorOps.Conv2D(input, weight, stride: 1, padding: 1);

            int iters = 5;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                var outConv = TensorOps.Conv2D(input, weight, stride: 1, padding: 1);
            }
            sw.Stop();
            double avgMs = sw.Elapsed.TotalMilliseconds / iters;

            Console.WriteLine($"Conv2D [1x3x64x64] -> [1x16x64x64]: {avgMs:F2} ms\n");
        }

        private static void RunZeroCopySliceBenchmark()
        {
            Console.WriteLine("--- 8. Zero-Copy View & Slicing Latency & Memory Allocation ---");
            var tensor = Tensor.Zeros<float>(1024, 1024);

            int iters = 100_000;

            // Measure GC bytes allocated during SubTensor / Slicing
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            long startBytes = GC.GetAllocatedBytesForCurrentThread();

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                var slice = tensor.SubTensor(500);
            }
            sw.Stop();
            long totalAllocated = GC.GetAllocatedBytesForCurrentThread() - startBytes;

            double nsPerSlice = (sw.Elapsed.TotalMilliseconds * 1e6) / iters;
            double bytesPerSlice = (double)totalAllocated / iters;

            Console.WriteLine($"SubTensor View Latency : {nsPerSlice:F1} ns/op");
            Console.WriteLine($"GC Allocation per View : {bytesPerSlice:F0} bytes (Header wrapper only, 0 data copy)\n");
        }

        private static void RunMemoryAllocationBenchmark()
        {
            Console.WriteLine("--- 9. Memory Allocation: Managed Array vs TensorPool ---");
            int iters = 10_000;

            // Standard Allocation
            GC.Collect();
            long startBytes = GC.GetAllocatedBytesForCurrentThread();
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                var t = new Tensor<float>(128, 128);
            }
            sw.Stop();
            long standardAlloc = GC.GetAllocatedBytesForCurrentThread() - startBytes;
            double stdMs = sw.Elapsed.TotalMilliseconds;

            // Pooled Allocation
            GC.Collect();
            startBytes = GC.GetAllocatedBytesForCurrentThread();
            sw.Restart();
            for (int i = 0; i < iters; i++)
            {
                using var pooled = TensorPool.Rent<float>(128, 128);
            }
            sw.Stop();
            long pooledAlloc = GC.GetAllocatedBytesForCurrentThread() - startBytes;
            double poolMs = sw.Elapsed.TotalMilliseconds;

            Console.WriteLine($"Standard `new Tensor` : {stdMs:F2} ms | Total GC Allocated: {standardAlloc / (1024.0 * 1024.0):F2} MB");
            Console.WriteLine($"TensorPool Rent/Return: {poolMs:F2} ms | Total GC Allocated: {pooledAlloc / (1024.0 * 1024.0):F2} MB | Speedup: {stdMs / poolMs:F2}x");
            Console.WriteLine($"GC Reduction Ratio    : {(double)standardAlloc / Math.Max(1, pooledAlloc):F1}x lower heap footprint\n");
        }
    }
}
