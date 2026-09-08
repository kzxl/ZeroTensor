using System;
using System.Numerics;
using System.Threading.Tasks;

namespace ZeroTensor.Core
{
    /// <summary>
    /// High-performance Basic Linear Algebra Subprograms (BLAS) engine.
    /// Provides cache-tiled SIMD General Matrix Multiplication (GEMM), batched matrix products, dot products, and vector norms.
    /// </summary>
    public static class TensorBlas
    {
        private const int BlockM = 32;
        private const int BlockN = 32;
        private const int BlockK = 32;

        /// <summary>
        /// Computes the matrix multiplication of two tensors: C = A @ B.
        /// Supports 1D x 2D, 2D x 1D, 2D x 2D, and batched N-D matrix multiplication.
        /// </summary>
        public static Tensor<float> MatMul(Tensor<float> a, Tensor<float> b)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));

            // Case 1: 1D x 2D
            if (a.Rank == 1 && b.Rank == 2)
            {
                var a2D = a.Unsqueeze(0); // [1, K]
                var res2D = MatMul2D(a2D, b); // [1, N]
                return res2D.Squeeze(0); // [N]
            }

            // Case 2: 2D x 1D
            if (a.Rank == 2 && b.Rank == 1)
            {
                var b2D = b.Unsqueeze(1); // [K, 1]
                var res2D = MatMul2D(a, b2D); // [M, 1]
                return res2D.Squeeze(1); // [M]
            }

            // Case 3: 2D x 2D
            if (a.Rank == 2 && b.Rank == 2)
            {
                return MatMul2D(a, b);
            }

            // Case 4: Batched Matrix Multiplication
            if (a.Rank >= 2 && b.Rank >= 2)
            {
                return BatchedMatMul(a, b);
            }

            throw new InvalidOperationException($"Cannot perform matrix multiplication on tensors with ranks {a.Rank} and {b.Rank}.");
        }

        /// <summary>
        /// Cache-tiled SIMD General Matrix Multiplication: C = A (M x K) @ B (K x N).
        /// </summary>
        public static Tensor<float> MatMul2D(Tensor<float> a, Tensor<float> b)
        {
            if (a.Rank != 2) throw new ArgumentException("Tensor A must have rank 2.", nameof(a));
            if (b.Rank != 2) throw new ArgumentException("Tensor B must have rank 2.", nameof(b));

            int m = a.Shape[0];
            int kA = a.Shape[1];
            int kB = b.Shape[0];
            int n = b.Shape[1];

            if (kA != kB)
            {
                throw new InvalidOperationException($"Inner matrix dimensions must agree: A is ({m}x{kA}), B is ({kB}x{n}).");
            }

            var c = new Tensor<float>(m, n);
            Gemm(a, b, c);
            return c;
        }

        /// <summary>
        /// Cache-tiled GEMM: C = alpha * (A @ B) + beta * C.
        /// Optimized with SIMD row-vectorization and multithreading over M-tiles.
        /// </summary>
        public static void Gemm(Tensor<float> a, Tensor<float> b, Tensor<float> c, float alpha = 1.0f, float beta = 0.0f)
        {
            int m = a.Shape[0];
            int k = a.Shape[1];
            int n = b.Shape[1];

            // If beta != 1, scale C first
            if (beta == 0.0f)
            {
                c.Fill(0f);
            }
            else if (beta != 1.0f)
            {
                var spanC = c.AsSpan();
                for (int i = 0; i < spanC.Length; i++) spanC[i] *= beta;
            }

            // Ensure contiguous or optimized access
            var aContig = a.ToContiguous();
            var bContig = b.ToContiguous();

            var bufA = aContig.Buffer;
            var bufB = bContig.Buffer;
            var bufC = c.Buffer;

            int offA = aContig.Offset;
            int offB = bContig.Offset;
            int offC = c.Offset;

            int sA0 = aContig.Strides[0], sA1 = aContig.Strides[1];
            int sB0 = bContig.Strides[0], sB1 = bContig.Strides[1];
            int sC0 = c.Strides[0], sC1 = c.Strides[1];

            int vecSize = Vector<float>.Count;

            // Multithread outer M-blocks for larger matrices
            bool runParallel = (m * n >= 4096) && (m >= BlockM * 2);

            Action<int> processMBlock = (m0) =>
            {
                int mEnd = Math.Min(m0 + BlockM, m);

                for (int k0 = 0; k0 < k; k0 += BlockK)
                {
                    int kEnd = Math.Min(k0 + BlockK, k);

                    for (int n0 = 0; n0 < n; n0 += BlockN)
                    {
                        int nEnd = Math.Min(n0 + BlockN, n);

                        // Inner micro-kernel
                        for (int i = m0; i < mEnd; i++)
                        {
                            int rowOffA = offA + i * sA0;
                            int rowOffC = offC + i * sC0;

                            for (int p = k0; p < kEnd; p++)
                            {
                                float aVal = bufA[rowOffA + p * sA1] * alpha;
                                if (aVal == 0f) continue;

                                var vA = new Vector<float>(aVal);
                                int rowOffB = offB + p * sB0;

                                int j = n0;
                                // SIMD vector loop along contiguous N dimension
                                for (; j <= nEnd - vecSize; j += vecSize)
                                {
                                    var vB = new Vector<float>(bufB, rowOffB + j * sB1);
                                    var vC = new Vector<float>(bufC, rowOffC + j * sC1);
                                    (vC + vA * vB).CopyTo(bufC, rowOffC + j * sC1);
                                }

                                // Scalar cleanup
                                for (; j < nEnd; j++)
                                {
                                    bufC[rowOffC + j * sC1] += aVal * bufB[rowOffB + j * sB1];
                                }
                            }
                        }
                    }
                }
            };

            if (runParallel)
            {
                int numMBlocks = (m + BlockM - 1) / BlockM;
                Parallel.For(0, numMBlocks, blockIdx =>
                {
                    processMBlock(blockIdx * BlockM);
                });
            }
            else
            {
                for (int m0 = 0; m0 < m; m0 += BlockM)
                {
                    processMBlock(m0);
                }
            }
        }

        private static Tensor<float> BatchedMatMul(Tensor<float> a, Tensor<float> b)
        {
            // Extract batch dimensions and matrix dimensions
            int aRank = a.Rank;
            int bRank = b.Rank;

            int m = a.Shape[aRank - 2];
            int kA = a.Shape[aRank - 1];
            int kB = b.Shape[bRank - 2];
            int n = b.Shape[bRank - 1];

            if (kA != kB)
            {
                throw new InvalidOperationException($"Batched matrix dimensions mismatch: A inner is {kA}, B inner is {kB}.");
            }

            // Extract batch shapes
            var aBatchDims = new int[aRank - 2];
            for (int i = 0; i < aRank - 2; i++) aBatchDims[i] = a.Shape[i];

            var bBatchDims = new int[bRank - 2];
            for (int i = 0; i < bRank - 2; i++) bBatchDims[i] = b.Shape[i];

            var aBatchShape = new TensorShape(aBatchDims);
            var bBatchShape = new TensorShape(bBatchDims);

            if (!aBatchShape.IsCompatibleForBroadcasting(bBatchShape, out var outBatchShape))
            {
                throw new InvalidOperationException($"Batch dimensions {aBatchShape} and {bBatchShape} cannot be broadcast.");
            }

            var outDims = new int[outBatchShape.Rank + 2];
            for (int i = 0; i < outBatchShape.Rank; i++) outDims[i] = outBatchShape[i];
            outDims[outDims.Length - 2] = m;
            outDims[outDims.Length - 1] = n;

            var result = new Tensor<float>(new TensorShape(outDims));
            int batchCount = outBatchShape.TotalElements;

            // Iterate over all batches
            var aBroadcast = a.BroadcastTo(new TensorShape(CombineBatchAndMatrix(outBatchShape, m, kA)));
            var bBroadcast = b.BroadcastTo(new TensorShape(CombineBatchAndMatrix(outBatchShape, kB, n)));

            int matASize = m * kA;
            int matBSize = kB * n;
            int matCSize = m * n;

            for (int bIdx = 0; bIdx < batchCount; bIdx++)
            {
                // Create sub-tensor 2D views for batch item
                var aSlice = aBroadcast.Slice(0, bIdx, 1).Squeeze(0);
                var bSlice = bBroadcast.Slice(0, bIdx, 1).Squeeze(0);
                var cSlice = result.Slice(0, bIdx, 1).Squeeze(0);

                Gemm(aSlice, bSlice, cSlice);
            }

            return result;
        }

        private static int[] CombineBatchAndMatrix(TensorShape batchShape, int r, int c)
        {
            var res = new int[batchShape.Rank + 2];
            for (int i = 0; i < batchShape.Rank; i++) res[i] = batchShape[i];
            res[res.Length - 2] = r;
            res[res.Length - 1] = c;
            return res;
        }

        /// <summary>
        /// Computes the dot product of two 1D vectors: sum(A[i] * B[i]).
        /// </summary>
        public static float Dot(Tensor<float> a, Tensor<float> b)
        {
            if (a.Rank != 1 || b.Rank != 1)
            {
                throw new ArgumentException($"Dot product requires 1D vectors, got ranks {a.Rank} and {b.Rank}.");
            }

            if (a.Length != b.Length)
            {
                throw new ArgumentException($"Vectors must have equal length: {a.Length} != {b.Length}.");
            }

            float dot = 0f;
            if (a.IsContiguous && b.IsContiguous)
            {
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int length = a.Length;

                int vecSize = Vector<float>.Count;
                var acc = Vector<float>.Zero;
                int i = 0;
                for (; i <= length - vecSize; i += vecSize)
                {
                    var va = new Vector<float>(bufA, offA + i);
                    var vb = new Vector<float>(bufB, offB + i);
                    acc += va * vb;
                }
                for (int v = 0; v < vecSize; v++) dot += acc[v];
                for (; i < length; i++) dot += bufA[offA + i] * bufB[offB + i];
                return dot;
            }

            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
            }
            return dot;
        }

        /// <summary>
        /// Computes the Euclidean / Frobenius norm of a tensor: sqrt(sum(x^2)).
        /// </summary>
        public static float Norm(Tensor<float> a)
        {
            float sumSq = 0f;
            if (a.IsContiguous)
            {
                var buf = a.Buffer;
                int off = a.Offset;
                int length = a.Length;

                int vecSize = Vector<float>.Count;
                var acc = Vector<float>.Zero;
                int i = 0;
                for (; i <= length - vecSize; i += vecSize)
                {
                    var v = new Vector<float>(buf, off + i);
                    acc += v * v;
                }
                for (int v = 0; v < vecSize; v++) sumSq += acc[v];
                for (; i < length; i++) sumSq += buf[off + i] * buf[off + i];
                return (float)Math.Sqrt(sumSq);
            }

            a.ForEachElement(v => sumSq += v * v);
            return (float)Math.Sqrt(sumSq);
        }
    }
}
