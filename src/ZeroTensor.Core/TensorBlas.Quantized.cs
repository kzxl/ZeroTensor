using System;
using System.Threading.Tasks;

namespace ZeroTensor.Core
{
    public static partial class TensorBlas
    {
        /// <summary>
        /// Computes matrix multiplication of two INT8 (sbyte) tensors: C (M x N) = A (M x K) @ B (K x N), accumulating into INT32.
        /// </summary>
        public static Tensor<int> MatMul(Tensor<sbyte> a, Tensor<sbyte> b)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (a.Rank != 2 || b.Rank != 2) throw new ArgumentException("INT8 MatMul currently supports 2D matrices.");

            int m = a.Shape[0];
            int k = a.Shape[1];
            int n = b.Shape[1];

            if (k != b.Shape[0])
            {
                throw new InvalidOperationException($"Inner matrix dimensions must agree: A is ({m}x{k}), B is ({b.Shape[0]}x{n}).");
            }

            var c = new Tensor<int>(m, n);
            GemmInt8(a, b, c);
            return c;
        }

        /// <summary>
        /// Cache-blocked INT8 General Matrix Multiply with INT32 accumulation: C = A @ B.
        /// </summary>
        public static void GemmInt8(Tensor<sbyte> a, Tensor<sbyte> b, Tensor<int> c)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (c == null) throw new ArgumentNullException(nameof(c));

            int m = a.Shape[0];
            int k = a.Shape[1];
            int n = b.Shape[1];

            c.Fill(0);

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

                        for (int i = m0; i < mEnd; i++)
                        {
                            int rowOffA = offA + i * sA0;
                            int rowOffC = offC + i * sC0;

                            for (int p = k0; p < kEnd; p++)
                            {
                                int aVal = bufA[rowOffA + p * sA1];
                                if (aVal == 0) continue;

                                int rowOffB = offB + p * sB0;
                                int j = n0;

                                // 4-way unrolled inner loop
                                for (; j <= nEnd - 4; j += 4)
                                {
                                    bufC[rowOffC + j * sC1] += aVal * bufB[rowOffB + j * sB1];
                                    bufC[rowOffC + (j + 1) * sC1] += aVal * bufB[rowOffB + (j + 1) * sB1];
                                    bufC[rowOffC + (j + 2) * sC1] += aVal * bufB[rowOffB + (j + 2) * sB1];
                                    bufC[rowOffC + (j + 3) * sC1] += aVal * bufB[rowOffB + (j + 3) * sB1];
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

        /// <summary>
        /// Quantized INT8 GEMM with zero-point offset and FP32 scale output:
        /// C_float[i, j] = scaleA * scaleB * sum((A[i, p] - zeroPointA) * (B[p, j] - zeroPointB)).
        /// </summary>
        public static void GemmInt8(
            Tensor<sbyte> a,
            Tensor<sbyte> b,
            Tensor<float> c,
            float scaleA = 1.0f,
            float scaleB = 1.0f,
            int zeroPointA = 0,
            int zeroPointB = 0)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (c == null) throw new ArgumentNullException(nameof(c));

            int m = a.Shape[0];
            int k = a.Shape[1];
            int n = b.Shape[1];

            float combinedScale = scaleA * scaleB;

            // First compute integer accumulation: C_int = sum((A - zA) * (B - zB))
            var intC = new Tensor<int>(m, n);

            var aContig = a.ToContiguous();
            var bContig = b.ToContiguous();

            var bufA = aContig.Buffer;
            var bufB = bContig.Buffer;
            var bufIntC = intC.Buffer;

            int offA = aContig.Offset;
            int offB = bContig.Offset;
            int offC = intC.Offset;

            int sA0 = aContig.Strides[0], sA1 = aContig.Strides[1];
            int sB0 = bContig.Strides[0], sB1 = bContig.Strides[1];
            int sC0 = intC.Strides[0], sC1 = intC.Strides[1];

            for (int i = 0; i < m; i++)
            {
                int rowOffA = offA + i * sA0;
                int rowOffC = offC + i * sC0;

                for (int p = 0; p < k; p++)
                {
                    int aVal = bufA[rowOffA + p * sA1] - zeroPointA;
                    if (aVal == 0) continue;

                    int rowOffB = offB + p * sB0;

                    for (int j = 0; j < n; j++)
                    {
                        int bVal = bufB[rowOffB + j * sB1] - zeroPointB;
                        bufIntC[rowOffC + j * sC1] += aVal * bVal;
                    }
                }
            }

            // Dequantize to float output
            var spanC = c.AsSpan();
            var spanIntC = intC.AsSpan();
            for (int i = 0; i < spanC.Length; i++)
            {
                spanC[i] = spanIntC[i] * combinedScale;
            }
        }

        /// <summary>
        /// Packed INT4 (4-bit nibble) General Matrix Multiply: C (M x N) = A (M x K) @ B_int4 (K x N).
        /// Each byte in packedWeights stores two 4-bit weights (low nibble = row 2*l, high nibble = row 2*l+1).
        /// Real weights are computed via: weight = (nibble - zeroPoint) * scale.
        /// Delivers 4x memory compression for edge LLM and vision model inference.
        /// </summary>
        public static Tensor<float> GemmInt4(
            Tensor<float> activation,
            Tensor<byte> packedWeights,
            Tensor<float> scales,
            Tensor<float>? zeroPoints = null)
        {
            if (activation == null) throw new ArgumentNullException(nameof(activation));
            if (packedWeights == null) throw new ArgumentNullException(nameof(packedWeights));
            if (scales == null) throw new ArgumentNullException(nameof(scales));

            if (activation.Rank != 2 || packedWeights.Rank != 2)
                throw new ArgumentException("GemmInt4 requires rank-2 activation and packed weight tensors.");

            int m = activation.Shape[0];
            int k = activation.Shape[1];
            int packedK = packedWeights.Shape[0];
            int n = packedWeights.Shape[1];

            if (packedK != (k + 1) / 2)
            {
                throw new InvalidOperationException($"Inner dimension mismatch: activation K={k}, but packedWeights K={packedK} (expected {(k + 1) / 2}).");
            }
            if (scales.Length < n)
            {
                throw new InvalidOperationException($"Scales length {scales.Length} must be at least N={n}.");
            }

            var c = new Tensor<float>(m, n);
            var actContig = activation.ToContiguous();
            var actBuf = actContig.Buffer;
            int actOff = actContig.Offset;
            int sAct0 = actContig.Strides[0], sAct1 = actContig.Strides[1];

            var wContig = packedWeights.ToContiguous();
            var wBuf = wContig.Buffer;
            int wOff = wContig.Offset;
            int sW0 = wContig.Strides[0], sW1 = wContig.Strides[1];

            var cBuf = c.Buffer;
            int cOff = c.Offset;
            int sC0 = c.Strides[0], sC1 = c.Strides[1];

            var sBuf = scales.Buffer;
            int sOff = scales.Offset;

            var zpBuf = zeroPoints?.Buffer;
            int zpOff = zeroPoints?.Offset ?? 0;

            Parallel.For(0, m, i =>
            {
                int actRowBase = actOff + i * sAct0;
                int cRowBase = cOff + i * sC0;

                for (int j = 0; j < n; j++)
                {
                    float scale = sBuf[sOff + j];
                    float zp = zpBuf != null ? zpBuf[zpOff + j] : 0f;
                    float acc = 0f;

                    for (int l = 0; l < packedK; l++)
                    {
                        byte packed = wBuf[wOff + l * sW0 + j * sW1];
                        int nibble0 = packed & 0x0F;
                        int nibble1 = (packed >> 4) & 0x0F;

                        int k0 = l * 2;
                        int k1 = k0 + 1;

                        float w0 = (nibble0 - zp) * scale;
                        acc += actBuf[actRowBase + k0 * sAct1] * w0;

                        if (k1 < k)
                        {
                            float w1 = (nibble1 - zp) * scale;
                            acc += actBuf[actRowBase + k1 * sAct1] * w1;
                        }
                    }

                    cBuf[cRowBase + j * sC1] = acc;
                }
            });

            return c;
        }
    }
}
