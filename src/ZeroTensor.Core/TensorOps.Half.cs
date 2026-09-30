using System;
using System.Runtime.CompilerServices;

namespace ZeroTensor.Core
{
    public static partial class TensorOps
    {
        #region Half-Precision Conversions

        /// <summary>
        /// Converts a single-precision float tensor to a half-precision (FP16) tensor.
        /// </summary>
        public static Tensor<Half> ToHalf(this Tensor<float> tensor)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));

            var result = new Tensor<Half>(tensor.Shape);
            var src = tensor.Buffer;
            var dst = result.Buffer;
            int length = tensor.Length;
            int srcOff = tensor.Offset;
            int dstOff = result.Offset;

            if (tensor.IsContiguous)
            {
                for (int i = 0; i < length; i++)
                {
                    dst[dstOff + i] = (Half)src[srcOff + i];
                }
            }
            else
            {
                tensor.Shape.ForEachCoordinate(coords =>
                {
                    result[coords] = (Half)tensor[coords];
                });
            }

            return result;
        }

        /// <summary>
        /// Converts a half-precision (FP16) tensor to a single-precision float tensor.
        /// </summary>
        public static Tensor<float> ToFloat(this Tensor<Half> tensor)
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
        /// Converts a double-precision tensor to a half-precision (FP16) tensor.
        /// </summary>
        public static Tensor<Half> ToHalf(this Tensor<double> tensor)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));

            var result = new Tensor<Half>(tensor.Shape);
            var src = tensor.Buffer;
            var dst = result.Buffer;
            int length = tensor.Length;
            int srcOff = tensor.Offset;
            int dstOff = result.Offset;

            if (tensor.IsContiguous)
            {
                for (int i = 0; i < length; i++)
                {
                    dst[dstOff + i] = (Half)(float)src[srcOff + i];
                }
            }
            else
            {
                tensor.Shape.ForEachCoordinate(coords =>
                {
                    result[coords] = (Half)(float)tensor[coords];
                });
            }

            return result;
        }

        /// <summary>
        /// Converts a half-precision (FP16) tensor to a double-precision tensor.
        /// </summary>
        public static Tensor<double> ToDouble(this Tensor<Half> tensor)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));

            var result = new Tensor<double>(tensor.Shape);
            var src = tensor.Buffer;
            var dst = result.Buffer;
            int length = tensor.Length;
            int srcOff = tensor.Offset;
            int dstOff = result.Offset;

            if (tensor.IsContiguous)
            {
                for (int i = 0; i < length; i++)
                {
                    dst[dstOff + i] = (double)(float)src[srcOff + i];
                }
            }
            else
            {
                tensor.Shape.ForEachCoordinate(coords =>
                {
                    result[coords] = (double)(float)tensor[coords];
                });
            }

            return result;
        }

        #endregion

        #region Binary Arithmetic Implementations for Half

        internal static Tensor<Half> AddHalf(Tensor<Half> a, Tensor<Half> b)
        {
            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var result = new Tensor<Half>(a.Shape);
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                var bufRes = result.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int offRes = result.Offset;
                int length = a.Length;

                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (Half)((float)bufA[offA + i] + (float)bufB[offB + i]);
                }
                return result;
            }

            return TensorBroadcaster.Apply(a, b, (x, y) => (Half)((float)x + (float)y));
        }

        internal static Tensor<Half> SubtractHalf(Tensor<Half> a, Tensor<Half> b)
        {
            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var result = new Tensor<Half>(a.Shape);
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                var bufRes = result.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int offRes = result.Offset;
                int length = a.Length;

                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (Half)((float)bufA[offA + i] - (float)bufB[offB + i]);
                }
                return result;
            }

            return TensorBroadcaster.Apply(a, b, (x, y) => (Half)((float)x - (float)y));
        }

        internal static Tensor<Half> MultiplyHalf(Tensor<Half> a, Tensor<Half> b)
        {
            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var result = new Tensor<Half>(a.Shape);
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                var bufRes = result.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int offRes = result.Offset;
                int length = a.Length;

                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (Half)((float)bufA[offA + i] * (float)bufB[offB + i]);
                }
                return result;
            }

            return TensorBroadcaster.Apply(a, b, (x, y) => (Half)((float)x * (float)y));
        }

        internal static Tensor<Half> DivideHalf(Tensor<Half> a, Tensor<Half> b)
        {
            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var result = new Tensor<Half>(a.Shape);
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                var bufRes = result.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int offRes = result.Offset;
                int length = a.Length;

                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (Half)((float)bufA[offA + i] / (float)bufB[offB + i]);
                }
                return result;
            }

            return TensorBroadcaster.Apply(a, b, (x, y) => (Half)((float)x / (float)y));
        }

        internal static Tensor<Half> NegateHalf(Tensor<Half> a)
        {
            var result = new Tensor<Half>(a.Shape);
            var bufA = a.Buffer;
            var bufRes = result.Buffer;
            int offA = a.Offset;
            int offRes = result.Offset;
            int length = a.Length;

            if (a.IsContiguous)
            {
                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (Half)(-(float)bufA[offA + i]);
                }
                return result;
            }

            a.Shape.ForEachCoordinate(coords =>
            {
                result[coords] = (Half)(-(float)a[coords]);
            });
            return result;
        }

        internal static Tensor<Half> AddScalarHalf(Tensor<Half> a, Half scalar)
        {
            var result = new Tensor<Half>(a.Shape);
            var bufA = a.Buffer;
            var bufRes = result.Buffer;
            int offA = a.Offset;
            int offRes = result.Offset;
            int length = a.Length;
            float s = (float)scalar;

            if (a.IsContiguous)
            {
                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (Half)((float)bufA[offA + i] + s);
                }
                return result;
            }

            a.Shape.ForEachCoordinate(coords =>
            {
                result[coords] = (Half)((float)a[coords] + s);
            });
            return result;
        }

        internal static Tensor<Half> SubtractScalarHalf(Tensor<Half> a, Half scalar)
        {
            var result = new Tensor<Half>(a.Shape);
            var bufA = a.Buffer;
            var bufRes = result.Buffer;
            int offA = a.Offset;
            int offRes = result.Offset;
            int length = a.Length;
            float s = (float)scalar;

            if (a.IsContiguous)
            {
                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (Half)((float)bufA[offA + i] - s);
                }
                return result;
            }

            a.Shape.ForEachCoordinate(coords =>
            {
                result[coords] = (Half)((float)a[coords] - s);
            });
            return result;
        }

        internal static Tensor<Half> MultiplyScalarHalf(Tensor<Half> a, Half scalar)
        {
            var result = new Tensor<Half>(a.Shape);
            var bufA = a.Buffer;
            var bufRes = result.Buffer;
            int offA = a.Offset;
            int offRes = result.Offset;
            int length = a.Length;
            float s = (float)scalar;

            if (a.IsContiguous)
            {
                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (Half)((float)bufA[offA + i] * s);
                }
                return result;
            }

            a.Shape.ForEachCoordinate(coords =>
            {
                result[coords] = (Half)((float)a[coords] * s);
            });
            return result;
        }

        internal static Tensor<Half> DivideScalarHalf(Tensor<Half> a, Half scalar)
        {
            var result = new Tensor<Half>(a.Shape);
            var bufA = a.Buffer;
            var bufRes = result.Buffer;
            int offA = a.Offset;
            int offRes = result.Offset;
            int length = a.Length;
            float s = (float)scalar;

            if (a.IsContiguous)
            {
                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (Half)((float)bufA[offA + i] / s);
                }
                return result;
            }

            a.Shape.ForEachCoordinate(coords =>
            {
                result[coords] = (Half)((float)a[coords] / s);
            });
            return result;
        }

        #endregion

        #region Unary Math & Activations for Half

        internal static Tensor<Half> AbsHalf(Tensor<Half> a) => ApplyUnaryHalf(a, x => Math.Abs(x));
        internal static Tensor<Half> ExpHalf(Tensor<Half> a) => ApplyUnaryHalf(a, x => (float)Math.Exp(x));
        internal static Tensor<Half> SqrtHalf(Tensor<Half> a) => ApplyUnaryHalf(a, x => (float)Math.Sqrt(x));
        internal static Tensor<Half> LogHalf(Tensor<Half> a) => ApplyUnaryHalf(a, x => (float)Math.Log(x));
        internal static Tensor<Half> SinHalf(Tensor<Half> a) => ApplyUnaryHalf(a, x => (float)Math.Sin(x));
        internal static Tensor<Half> CosHalf(Tensor<Half> a) => ApplyUnaryHalf(a, x => (float)Math.Cos(x));
        internal static Tensor<Half> TanhHalf(Tensor<Half> a) => ApplyUnaryHalf(a, x => (float)Math.Tanh(x));

        internal static Tensor<Half> ReLUHalf(Tensor<Half> a) =>
            ApplyUnaryHalf(a, x => x > 0f ? x : 0f);

        internal static Tensor<Half> GELUHalf(Tensor<Half> a) =>
            ApplyUnaryHalf(a, x => 0.5f * x * (1.0f + (float)Math.Tanh(0.7978845608f * (x + 0.044715f * x * x * x))));

        internal static Tensor<Half> SigmoidHalf(Tensor<Half> a) =>
            ApplyUnaryHalf(a, x => 1.0f / (1.0f + (float)Math.Exp(-x)));

        private static Tensor<Half> ApplyUnaryHalf(Tensor<Half> a, Func<float, float> func)
        {
            var result = new Tensor<Half>(a.Shape);
            var bufA = a.Buffer;
            var bufRes = result.Buffer;
            int offA = a.Offset;
            int offRes = result.Offset;
            int length = a.Length;

            if (a.IsContiguous)
            {
                for (int i = 0; i < length; i++)
                {
                    bufRes[offRes + i] = (Half)func((float)bufA[offA + i]);
                }
                return result;
            }

            a.Shape.ForEachCoordinate(coords =>
            {
                result[coords] = (Half)func((float)a[coords]);
            });
            return result;
        }

        #endregion

        #region Reductions for Half

        internal static Half SumHalf(Tensor<Half> a)
        {
            float acc = 0f;
            var buf = a.Buffer;
            int off = a.Offset;
            int len = a.Length;

            if (a.IsContiguous)
            {
                for (int i = 0; i < len; i++) acc += (float)buf[off + i];
                return (Half)acc;
            }

            a.Shape.ForEachCoordinate(coords => acc += (float)a[coords]);
            return (Half)acc;
        }

        internal static Half MeanHalf(Tensor<Half> a)
        {
            if (a.Length == 0) return (Half)0f;
            return (Half)((float)SumHalf(a) / a.Length);
        }

        internal static Half MaxHalf(Tensor<Half> a)
        {
            if (a.Length == 0) throw new InvalidOperationException("Cannot compute max on empty tensor.");
            float max = float.NegativeInfinity;
            var buf = a.Buffer;
            int off = a.Offset;
            int len = a.Length;

            if (a.IsContiguous)
            {
                for (int i = 0; i < len; i++)
                {
                    float v = (float)buf[off + i];
                    if (v > max) max = v;
                }
                return (Half)max;
            }

            a.Shape.ForEachCoordinate(coords =>
            {
                float v = (float)a[coords];
                if (v > max) max = v;
            });
            return (Half)max;
        }

        internal static Half MinHalf(Tensor<Half> a)
        {
            if (a.Length == 0) throw new InvalidOperationException("Cannot compute min on empty tensor.");
            float min = float.PositiveInfinity;
            var buf = a.Buffer;
            int off = a.Offset;
            int len = a.Length;

            if (a.IsContiguous)
            {
                for (int i = 0; i < len; i++)
                {
                    float v = (float)buf[off + i];
                    if (v < min) min = v;
                }
                return (Half)min;
            }

            a.Shape.ForEachCoordinate(coords =>
            {
                float v = (float)a[coords];
                if (v < min) min = v;
            });
            return (Half)min;
        }

        #endregion
    }
}
