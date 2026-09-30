using System;
using System.Numerics;
using System.Threading.Tasks;

namespace ZeroTensor.Core
{
    public static partial class TensorBlas
    {
        /// <summary>
        /// Computes the matrix multiplication of two double tensors: C = A @ B.
        /// Supports 1D x 2D, 2D x 1D, 2D x 2D, and batched N-D matrix multiplication.
        /// </summary>
        public static Tensor<double> MatMul(Tensor<double> a, Tensor<double> b)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));

            if (a.Rank == 1 && b.Rank == 2)
            {
                var a2D = a.Unsqueeze(0);
                var res2D = MatMul2D(a2D, b);
                return res2D.Squeeze(0);
            }

            if (a.Rank == 2 && b.Rank == 1)
            {
                var b2D = b.Unsqueeze(1);
                var res2D = MatMul2D(a, b2D);
                return res2D.Squeeze(1);
            }

            if (a.Rank == 2 && b.Rank == 2)
            {
                return MatMul2D(a, b);
            }

            if (a.Rank >= 2 && b.Rank >= 2)
            {
                return BatchedMatMul(a, b);
            }

            throw new InvalidOperationException($"Cannot perform matrix multiplication on tensors with ranks {a.Rank} and {b.Rank}.");
        }

        /// <summary>
        /// Cache-tiled SIMD General Matrix Multiplication for double precision: C = A (M x K) @ B (K x N).
        /// </summary>
        public static Tensor<double> MatMul2D(Tensor<double> a, Tensor<double> b)
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

            var c = new Tensor<double>(m, n);
            Gemm(a, b, c);
            return c;
        }

        /// <summary>
        /// Cache-tiled SIMD General Matrix Multiplication returning a new tensor (double): C = A @ B.
        /// </summary>
        public static Tensor<double> Gemm(Tensor<double> a, Tensor<double> b) => MatMul2D(a, b);

        /// <summary>
        /// Cache-tiled GEMM for double precision: C = alpha * (A @ B) + beta * C.
        /// </summary>
        public static void Gemm(Tensor<double> a, Tensor<double> b, Tensor<double> c, double alpha = 1.0, double beta = 0.0)
        {
            int m = a.Shape[0];
            int k = a.Shape[1];
            int n = b.Shape[1];

            if (beta == 0.0)
            {
                c.Fill(0.0);
            }
            else if (beta != 1.0)
            {
                var spanC = c.AsSpan();
                for (int i = 0; i < spanC.Length; i++) spanC[i] *= beta;
            }

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

            int vecSize = Vector<double>.Count;
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

                        int i = m0;
                        for (; i <= mEnd - 2; i += 2)
                        {
                            int rowOffA0 = offA + i * sA0;
                            int rowOffA1 = offA + (i + 1) * sA0;
                            int rowOffC0 = offC + i * sC0;
                            int rowOffC1 = offC + (i + 1) * sC0;

                            for (int p = k0; p < kEnd; p++)
                            {
                                double a0 = bufA[rowOffA0 + p * sA1] * alpha;
                                double a1 = bufA[rowOffA1 + p * sA1] * alpha;
                                if (a0 == 0.0 && a1 == 0.0) continue;

                                var vA0 = new Vector<double>(a0);
                                var vA1 = new Vector<double>(a1);
                                int rowOffB = offB + p * sB0;

                                int j = n0;
                                for (; j <= nEnd - vecSize; j += vecSize)
                                {
                                    var vB = new Vector<double>(bufB, rowOffB + j * sB1);
                                    var vC0 = new Vector<double>(bufC, rowOffC0 + j * sC1);
                                    var vC1 = new Vector<double>(bufC, rowOffC1 + j * sC1);

                                    (vC0 + vA0 * vB).CopyTo(bufC, rowOffC0 + j * sC1);
                                    (vC1 + vA1 * vB).CopyTo(bufC, rowOffC1 + j * sC1);
                                }

                                for (; j < nEnd; j++)
                                {
                                    double bVal = bufB[rowOffB + j * sB1];
                                    bufC[rowOffC0 + j * sC1] += a0 * bVal;
                                    bufC[rowOffC1 + j * sC1] += a1 * bVal;
                                }
                            }
                        }

                        // Tail cleanup for remaining odd row
                        for (; i < mEnd; i++)
                        {
                            int rowOffA = offA + i * sA0;
                            int rowOffC = offC + i * sC0;

                            for (int p = k0; p < kEnd; p++)
                            {
                                double aVal = bufA[rowOffA + p * sA1] * alpha;
                                if (aVal == 0.0) continue;

                                var vA = new Vector<double>(aVal);
                                int rowOffB = offB + p * sB0;

                                int j = n0;
                                for (; j <= nEnd - vecSize; j += vecSize)
                                {
                                    var vB = new Vector<double>(bufB, rowOffB + j * sB1);
                                    var vC = new Vector<double>(bufC, rowOffC + j * sC1);
                                    (vC + vA * vB).CopyTo(bufC, rowOffC + j * sC1);
                                }

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

        private static Tensor<double> BatchedMatMul(Tensor<double> a, Tensor<double> b)
        {
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

            var result = new Tensor<double>(new TensorShape(outDims));
            int batchCount = outBatchShape.TotalElements;

            var aBroadcast = a.BroadcastTo(new TensorShape(CombineBatchAndMatrix(outBatchShape, m, kA)));
            var bBroadcast = b.BroadcastTo(new TensorShape(CombineBatchAndMatrix(outBatchShape, kB, n)));

            var a3D = aBroadcast.Reshape(batchCount, m, kA);
            var b3D = bBroadcast.Reshape(batchCount, kB, n);
            var c3D = result.Reshape(batchCount, m, n);

            for (int bIdx = 0; bIdx < batchCount; bIdx++)
            {
                var aSlice = a3D.SubTensor(bIdx);
                var bSlice = b3D.SubTensor(bIdx);
                var cSlice = c3D.SubTensor(bIdx);

                Gemm(aSlice, bSlice, cSlice);
            }

            return result;
        }

        /// <summary>
        /// Computes the dot product of two 1D double vectors: sum(A[i] * B[i]).
        /// </summary>
        public static double Dot(Tensor<double> a, Tensor<double> b)
        {
            if (a.Rank != 1 || b.Rank != 1)
            {
                throw new ArgumentException($"Dot product requires 1D vectors, got ranks {a.Rank} and {b.Rank}.");
            }

            if (a.Length != b.Length)
            {
                throw new ArgumentException($"Vectors must have equal length: {a.Length} != {b.Length}.");
            }

            double dot = 0.0;
            if (a.IsContiguous && b.IsContiguous)
            {
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int length = a.Length;

                int vecSize = Vector<double>.Count;
                var acc = Vector<double>.Zero;
                int i = 0;
                for (; i <= length - vecSize; i += vecSize)
                {
                    var va = new Vector<double>(bufA, offA + i);
                    var vb = new Vector<double>(bufB, offB + i);
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
        /// Computes the Euclidean / Frobenius norm of a double tensor: sqrt(sum(x^2)).
        /// </summary>
        public static double Norm(Tensor<double> a)
        {
            double sumSq = 0.0;
            if (a.IsContiguous)
            {
                var buf = a.Buffer;
                int off = a.Offset;
                int length = a.Length;

                int vecSize = Vector<double>.Count;
                var acc = Vector<double>.Zero;
                int i = 0;
                for (; i <= length - vecSize; i += vecSize)
                {
                    var v = new Vector<double>(buf, off + i);
                    acc += v * v;
                }
                for (int v = 0; v < vecSize; v++) sumSq += acc[v];
                for (; i < length; i++) sumSq += buf[off + i] * buf[off + i];
                return Math.Sqrt(sumSq);
            }

            a.ForEachElement(v => sumSq += v * v);
            return Math.Sqrt(sumSq);
        }
    }
}
