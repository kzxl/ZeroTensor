using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace ZeroTensor.Core
{
    public static partial class TensorOps
    {
        #region BFloat16 Conversions

        /// <summary>
        /// Converts a single-precision float tensor to a Brain Floating Point (BFloat16) tensor.
        /// </summary>
        public static Tensor<BFloat16> ToBFloat16(this Tensor<float> tensor)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));

            var result = new Tensor<BFloat16>(tensor.Shape);
            var src = tensor.Buffer;
            var dst = result.Buffer;
            int length = tensor.Length;
            int srcOff = tensor.Offset;
            int dstOff = result.Offset;

            if (tensor.IsContiguous)
            {
                for (int i = 0; i < length; i++)
                {
                    dst[dstOff + i] = (BFloat16)src[srcOff + i];
                }
            }
            else
            {
                tensor.Shape.ForEachCoordinate(coords =>
                {
                    result[coords] = (BFloat16)tensor[coords];
                });
            }

            return result;
        }

        /// <summary>
        /// Converts a BFloat16 tensor to a single-precision float tensor.
        /// </summary>
        public static Tensor<float> ToFloat(this Tensor<BFloat16> tensor)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));

            var result = new Tensor<float>(tensor.Shape);
            var src = tensor.Buffer;
            var dst = result.Buffer;
            int length = tensor.Length;
            int srcOff = tensor.Offset;
            int dstOff = result.Offset;

            if (tensor.IsContiguous)
            {
                for (int i = 0; i < length; i++)
                {
                    dst[dstOff + i] = (float)src[srcOff + i];
                }
            }
            else
            {
                tensor.Shape.ForEachCoordinate(coords =>
                {
                    result[coords] = (float)tensor[coords];
                });
            }

            return result;
        }

        /// <summary>
        /// Converts an FP16 Half tensor to a BFloat16 tensor.
        /// </summary>
        public static Tensor<BFloat16> ToBFloat16(this Tensor<Half> tensor)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));

            var result = new Tensor<BFloat16>(tensor.Shape);
            var src = tensor.Buffer;
            var dst = result.Buffer;
            int length = tensor.Length;
            int srcOff = tensor.Offset;
            int dstOff = result.Offset;

            if (tensor.IsContiguous)
            {
                for (int i = 0; i < length; i++)
                {
                    dst[dstOff + i] = (BFloat16)src[srcOff + i];
                }
            }
            else
            {
                tensor.Shape.ForEachCoordinate(coords =>
                {
                    result[coords] = (BFloat16)tensor[coords];
                });
            }

            return result;
        }

        #endregion

        #region BFloat16 Binary Arithmetic

        internal static Tensor<BFloat16> AddBFloat16(Tensor<BFloat16> a, Tensor<BFloat16> b)
        {
            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var result = new Tensor<BFloat16>(a.Shape);
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                var bufRes = result.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int offRes = result.Offset;
                int length = a.Length;

                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (BFloat16)((float)bufA[offA + i] + (float)bufB[offB + i]);
                }
                return result;
            }

            return TensorBroadcaster.ApplyGeneric(a, b, (x, y) => (BFloat16)((float)x + (float)y));
        }

        internal static Tensor<BFloat16> SubtractBFloat16(Tensor<BFloat16> a, Tensor<BFloat16> b)
        {
            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var result = new Tensor<BFloat16>(a.Shape);
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                var bufRes = result.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int offRes = result.Offset;
                int length = a.Length;

                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (BFloat16)((float)bufA[offA + i] - (float)bufB[offB + i]);
                }
                return result;
            }

            return TensorBroadcaster.ApplyGeneric(a, b, (x, y) => (BFloat16)((float)x - (float)y));
        }

        internal static Tensor<BFloat16> MultiplyBFloat16(Tensor<BFloat16> a, Tensor<BFloat16> b)
        {
            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var result = new Tensor<BFloat16>(a.Shape);
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                var bufRes = result.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int offRes = result.Offset;
                int length = a.Length;

                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (BFloat16)((float)bufA[offA + i] * (float)bufB[offB + i]);
                }
                return result;
            }

            return TensorBroadcaster.ApplyGeneric(a, b, (x, y) => (BFloat16)((float)x * (float)y));
        }

        internal static Tensor<BFloat16> DivideBFloat16(Tensor<BFloat16> a, Tensor<BFloat16> b)
        {
            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var result = new Tensor<BFloat16>(a.Shape);
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                var bufRes = result.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int offRes = result.Offset;
                int length = a.Length;

                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (BFloat16)((float)bufA[offA + i] / (float)bufB[offB + i]);
                }
                return result;
            }

            return TensorBroadcaster.ApplyGeneric(a, b, (x, y) => (BFloat16)((float)x / (float)y));
        }

        internal static Tensor<BFloat16> NegateBFloat16(Tensor<BFloat16> a)
        {
            var result = new Tensor<BFloat16>(a.Shape);
            var bufA = a.Buffer;
            var bufRes = result.Buffer;
            int offA = a.Offset;
            int offRes = result.Offset;
            int length = a.Length;

            if (a.IsContiguous)
            {
                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (BFloat16)(-(float)bufA[offA + i]);
                }
                return result;
            }

            a.Shape.ForEachCoordinate(coords =>
            {
                result[coords] = (BFloat16)(-(float)a[coords]);
            });
            return result;
        }

        #endregion

        #region BFloat16 Activations & Reductions

        public static Tensor<BFloat16> ReLUBFloat16(Tensor<BFloat16> a)
        {
            var result = new Tensor<BFloat16>(a.Shape);
            var bufA = a.Buffer;
            var bufRes = result.Buffer;
            int offA = a.Offset;
            int offRes = result.Offset;
            int length = a.Length;

            if (a.IsContiguous)
            {
                for (int i = 0; i < length; i++)
                {
                    float v = (float)bufA[offA + i];
                    bufRes[offRes + i] = (BFloat16)(v > 0f ? v : 0f);
                }
                return result;
            }

            a.Shape.ForEachCoordinate(coords =>
            {
                float v = (float)a[coords];
                result[coords] = (BFloat16)(v > 0f ? v : 0f);
            });
            return result;
        }

        public static BFloat16 SumBFloat16(Tensor<BFloat16> a)
        {
            float acc = 0f;
            var buf = a.Buffer;
            int off = a.Offset;
            int len = a.Length;

            if (a.IsContiguous)
            {
                for (int i = 0; i < len; i++) acc += (float)buf[off + i];
                return (BFloat16)acc;
            }

            a.Shape.ForEachCoordinate(coords => acc += (float)a[coords]);
            return (BFloat16)acc;
        }

        #endregion
    }

    public static partial class TensorBlas
    {
        /// <summary>
        /// Cache-tiled matrix multiplication for BFloat16 tensors with FP32 register accumulation: C = A @ B.
        /// </summary>
        public static Tensor<BFloat16> MatMul(Tensor<BFloat16> a, Tensor<BFloat16> b)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (a.Rank != 2 || b.Rank != 2) throw new ArgumentException("BFloat16 MatMul currently supports 2D matrices.");

            int m = a.Shape[0];
            int kA = a.Shape[1];
            int kB = b.Shape[0];
            int n = b.Shape[1];

            if (kA != kB)
            {
                throw new InvalidOperationException($"Inner matrix dimensions must agree: A is ({m}x{kA}), B is ({kB}x{n}).");
            }

            int k = kA;
            var c = new Tensor<BFloat16>(m, n);

            var aBuf = a.Buffer;
            var bBuf = b.Buffer;
            var cBuf = c.Buffer;

            int aStride0 = a.Strides[0], aStride1 = a.Strides[1];
            int bStride0 = b.Strides[0], bStride1 = b.Strides[1];
            int cStride0 = c.Strides[0], cStride1 = c.Strides[1];

            int aOff = a.Offset;
            int bOff = b.Offset;
            int cOff = c.Offset;

            const int bm = 32;
            const int bn = 32;
            const int bk = 32;

            int numBlocksM = (m + bm - 1) / bm;

            if (m * n >= 4096)
            {
                Parallel.For(0, numBlocksM, blockMIdx =>
                {
                    int iStart = blockMIdx * bm;
                    int iEnd = Math.Min(iStart + bm, m);

                    for (int jStart = 0; jStart < n; jStart += bn)
                    {
                        int jEnd = Math.Min(jStart + bn, n);

                        for (int lStart = 0; lStart < k; lStart += bk)
                        {
                            int lEnd = Math.Min(lStart + bk, k);

                            for (int i = iStart; i < iEnd; i++)
                            {
                                int aRowBase = aOff + i * aStride0;
                                int cRowBase = cOff + i * cStride0;

                                for (int l = lStart; l < lEnd; l++)
                                {
                                    float aVal = (float)aBuf[aRowBase + l * aStride1];
                                    int bRowBase = bOff + l * bStride0;

                                    for (int j = jStart; j < jEnd; j++)
                                    {
                                        int cIdx = cRowBase + j * cStride1;
                                        float current = (float)cBuf[cIdx];
                                        cBuf[cIdx] = (BFloat16)(current + aVal * (float)bBuf[bRowBase + j * bStride1]);
                                    }
                                }
                            }
                        }
                    }
                });
            }
            else
            {
                for (int iStart = 0; iStart < m; iStart += bm)
                {
                    int iEnd = Math.Min(iStart + bm, m);

                    for (int jStart = 0; jStart < n; jStart += bn)
                    {
                        int jEnd = Math.Min(jStart + bn, n);

                        for (int lStart = 0; lStart < k; lStart += bk)
                        {
                            int lEnd = Math.Min(lStart + bk, k);

                            for (int i = iStart; i < iEnd; i++)
                            {
                                int aRowBase = aOff + i * aStride0;
                                int cRowBase = cOff + i * cStride0;

                                for (int l = lStart; l < lEnd; l++)
                                {
                                    float aVal = (float)aBuf[aRowBase + l * aStride1];
                                    int bRowBase = bOff + l * bStride0;

                                    for (int j = jStart; j < jEnd; j++)
                                    {
                                        int cIdx = cRowBase + j * cStride1;
                                        float current = (float)cBuf[cIdx];
                                        cBuf[cIdx] = (BFloat16)(current + aVal * (float)bBuf[bRowBase + j * bStride1]);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return c;
        }
    }
}
