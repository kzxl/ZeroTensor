using System;
using System.Numerics;

namespace ZeroTensor.Core
{
    public static partial class TensorOps
    {
        #region Float In-Place Arithmetic

        /// <summary>
        /// Adds tensor b into tensor a in-place: a += b.
        /// </summary>
        public static Tensor<float> Add_(this Tensor<float> a, Tensor<float> b)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));

            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int len = a.Length;
                int vecSize = Vector<float>.Count;
                int i = 0;

                for (; i <= len - vecSize; i += vecSize)
                {
                    var va = new Vector<float>(bufA, offA + i);
                    var vb = new Vector<float>(bufB, offB + i);
                    (va + vb).CopyTo(bufA, offA + i);
                }
                for (; i < len; i++)
                {
                    bufA[offA + i] += bufB[offB + i];
                }
                return a;
            }

            // General broadcasting in-place
            var bBroad = b.BroadcastTo(a.Shape);
            bBroad.ForEachCoordinate(coords => a[coords] += bBroad[coords]);
            return a;
        }

        /// <summary>
        /// Subtracts tensor b from tensor a in-place: a -= b.
        /// </summary>
        public static Tensor<float> Subtract_(this Tensor<float> a, Tensor<float> b)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));

            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int len = a.Length;
                int vecSize = Vector<float>.Count;
                int i = 0;

                for (; i <= len - vecSize; i += vecSize)
                {
                    var va = new Vector<float>(bufA, offA + i);
                    var vb = new Vector<float>(bufB, offB + i);
                    (va - vb).CopyTo(bufA, offA + i);
                }
                for (; i < len; i++)
                {
                    bufA[offA + i] -= bufB[offB + i];
                }
                return a;
            }

            var bBroad = b.BroadcastTo(a.Shape);
            bBroad.ForEachCoordinate(coords => a[coords] -= bBroad[coords]);
            return a;
        }

        /// <summary>
        /// Multiplies tensor a by tensor b element-wise in-place: a *= b.
        /// </summary>
        public static Tensor<float> Multiply_(this Tensor<float> a, Tensor<float> b)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));

            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int len = a.Length;
                int vecSize = Vector<float>.Count;
                int i = 0;

                for (; i <= len - vecSize; i += vecSize)
                {
                    var va = new Vector<float>(bufA, offA + i);
                    var vb = new Vector<float>(bufB, offB + i);
                    (va * vb).CopyTo(bufA, offA + i);
                }
                for (; i < len; i++)
                {
                    bufA[offA + i] *= bufB[offB + i];
                }
                return a;
            }

            var bBroad = b.BroadcastTo(a.Shape);
            bBroad.ForEachCoordinate(coords => a[coords] *= bBroad[coords]);
            return a;
        }

        /// <summary>
        /// Divides tensor a by tensor b element-wise in-place: a /= b.
        /// </summary>
        public static Tensor<float> Divide_(this Tensor<float> a, Tensor<float> b)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));

            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int len = a.Length;
                int vecSize = Vector<float>.Count;
                int i = 0;

                for (; i <= len - vecSize; i += vecSize)
                {
                    var va = new Vector<float>(bufA, offA + i);
                    var vb = new Vector<float>(bufB, offB + i);
                    (va / vb).CopyTo(bufA, offA + i);
                }
                for (; i < len; i++)
                {
                    bufA[offA + i] /= bufB[offB + i];
                }
                return a;
            }

            var bBroad = b.BroadcastTo(a.Shape);
            bBroad.ForEachCoordinate(coords => a[coords] /= bBroad[coords]);
            return a;
        }

        /// <summary>
        /// Adds a scalar to tensor a in-place: a += scalar.
        /// </summary>
        public static Tensor<float> AddScalar_(this Tensor<float> a, float scalar)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (a.IsContiguous)
            {
                var bufA = a.Buffer;
                int offA = a.Offset;
                int len = a.Length;
                var vScalar = new Vector<float>(scalar);
                int vecSize = Vector<float>.Count;
                int i = 0;
                for (; i <= len - vecSize; i += vecSize)
                {
                    var v = new Vector<float>(bufA, offA + i);
                    (v + vScalar).CopyTo(bufA, offA + i);
                }
                for (; i < len; i++) bufA[offA + i] += scalar;
                return a;
            }

            a.ForEachCoordinate(coords => a[coords] += scalar);
            return a;
        }

        /// <summary>
        /// Multiplies tensor a by a scalar in-place: a *= scalar.
        /// </summary>
        public static Tensor<float> MultiplyScalar_(this Tensor<float> a, float scalar)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (a.IsContiguous)
            {
                var bufA = a.Buffer;
                int offA = a.Offset;
                int len = a.Length;
                var vScalar = new Vector<float>(scalar);
                int vecSize = Vector<float>.Count;
                int i = 0;
                for (; i <= len - vecSize; i += vecSize)
                {
                    var v = new Vector<float>(bufA, offA + i);
                    (v * vScalar).CopyTo(bufA, offA + i);
                }
                for (; i < len; i++) bufA[offA + i] *= scalar;
                return a;
            }

            a.ForEachCoordinate(coords => a[coords] *= scalar);
            return a;
        }

        /// <summary>
        /// Applies ReLU activation in-place: a = max(0, a).
        /// </summary>
        public static Tensor<float> Relu_(this Tensor<float> a)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (a.IsContiguous)
            {
                var bufA = a.Buffer;
                int offA = a.Offset;
                int len = a.Length;
                var zeroVec = Vector<float>.Zero;
                int vecSize = Vector<float>.Count;
                int i = 0;
                for (; i <= len - vecSize; i += vecSize)
                {
                    var v = new Vector<float>(bufA, offA + i);
                    Vector.Max(v, zeroVec).CopyTo(bufA, offA + i);
                }
                for (; i < len; i++) bufA[offA + i] = Math.Max(0f, bufA[offA + i]);
                return a;
            }

            a.ForEachCoordinate(coords => a[coords] = Math.Max(0f, a[coords]));
            return a;
        }

        /// <summary>
        /// Clamps all elements in-place: a = clamp(a, min, max).
        /// </summary>
        public static Tensor<float> Clamp_(this Tensor<float> a, float min, float max)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (a.IsContiguous)
            {
                var bufA = a.Buffer;
                int offA = a.Offset;
                int len = a.Length;
                for (int i = 0; i < len; i++)
                {
                    float val = bufA[offA + i];
                    bufA[offA + i] = val < min ? min : (val > max ? max : val);
                }
                return a;
            }

            a.ForEachCoordinate(coords =>
            {
                float val = a[coords];
                a[coords] = val < min ? min : (val > max ? max : val);
            });
            return a;
        }

        #endregion

        #region Float Destination Buffer Operations

        /// <summary>
        /// Computes destination = a + b without allocating a new tensor buffer.
        /// </summary>
        public static void Add(Tensor<float> a, Tensor<float> b, Tensor<float> destination)
        {
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (a.Shape == b.Shape && a.Shape == destination.Shape && a.IsContiguous && b.IsContiguous && destination.IsContiguous)
            {
                var bufA = a.Buffer; int offA = a.Offset;
                var bufB = b.Buffer; int offB = b.Offset;
                var bufDst = destination.Buffer; int offDst = destination.Offset;
                int len = a.Length;
                int vecSize = Vector<float>.Count;
                int i = 0;

                for (; i <= len - vecSize; i += vecSize)
                {
                    var va = new Vector<float>(bufA, offA + i);
                    var vb = new Vector<float>(bufB, offB + i);
                    (va + vb).CopyTo(bufDst, offDst + i);
                }
                for (; i < len; i++)
                {
                    bufDst[offDst + i] = bufA[offA + i] + bufB[offB + i];
                }
                return;
            }

            var res = a + b;
            res.CopyTo(destination);
        }

        /// <summary>
        /// Computes destination = a - b without allocating a new tensor buffer.
        /// </summary>
        public static void Subtract(Tensor<float> a, Tensor<float> b, Tensor<float> destination)
        {
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (a.Shape == b.Shape && a.Shape == destination.Shape && a.IsContiguous && b.IsContiguous && destination.IsContiguous)
            {
                var bufA = a.Buffer; int offA = a.Offset;
                var bufB = b.Buffer; int offB = b.Offset;
                var bufDst = destination.Buffer; int offDst = destination.Offset;
                int len = a.Length;
                int vecSize = Vector<float>.Count;
                int i = 0;

                for (; i <= len - vecSize; i += vecSize)
                {
                    var va = new Vector<float>(bufA, offA + i);
                    var vb = new Vector<float>(bufB, offB + i);
                    (va - vb).CopyTo(bufDst, offDst + i);
                }
                for (; i < len; i++)
                {
                    bufDst[offDst + i] = bufA[offA + i] - bufB[offB + i];
                }
                return;
            }

            var res = a - b;
            res.CopyTo(destination);
        }

        /// <summary>
        /// Computes destination = a * b without allocating a new tensor buffer.
        /// </summary>
        public static void Multiply(Tensor<float> a, Tensor<float> b, Tensor<float> destination)
        {
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (a.Shape == b.Shape && a.Shape == destination.Shape && a.IsContiguous && b.IsContiguous && destination.IsContiguous)
            {
                var bufA = a.Buffer; int offA = a.Offset;
                var bufB = b.Buffer; int offB = b.Offset;
                var bufDst = destination.Buffer; int offDst = destination.Offset;
                int len = a.Length;
                int vecSize = Vector<float>.Count;
                int i = 0;

                for (; i <= len - vecSize; i += vecSize)
                {
                    var va = new Vector<float>(bufA, offA + i);
                    var vb = new Vector<float>(bufB, offB + i);
                    (va * vb).CopyTo(bufDst, offDst + i);
                }
                for (; i < len; i++)
                {
                    bufDst[offDst + i] = bufA[offA + i] * bufB[offB + i];
                }
                return;
            }

            var res = a * b;
            res.CopyTo(destination);
        }

        #endregion

        #region Double In-Place Arithmetic

        /// <summary>
        /// Adds tensor b into tensor a in-place: a += b (double precision).
        /// </summary>
        public static Tensor<double> Add_(this Tensor<double> a, Tensor<double> b)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));

            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int len = a.Length;
                int vecSize = Vector<double>.Count;
                int i = 0;

                for (; i <= len - vecSize; i += vecSize)
                {
                    var va = new Vector<double>(bufA, offA + i);
                    var vb = new Vector<double>(bufB, offB + i);
                    (va + vb).CopyTo(bufA, offA + i);
                }
                for (; i < len; i++)
                {
                    bufA[offA + i] += bufB[offB + i];
                }
                return a;
            }

            var bBroad = b.BroadcastTo(a.Shape);
            bBroad.ForEachCoordinate(coords => a[coords] += bBroad[coords]);
            return a;
        }

        /// <summary>
        /// Subtracts tensor b from tensor a in-place: a -= b (double precision).
        /// </summary>
        public static Tensor<double> Subtract_(this Tensor<double> a, Tensor<double> b)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));

            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int len = a.Length;
                int vecSize = Vector<double>.Count;
                int i = 0;

                for (; i <= len - vecSize; i += vecSize)
                {
                    var va = new Vector<double>(bufA, offA + i);
                    var vb = new Vector<double>(bufB, offB + i);
                    (va - vb).CopyTo(bufA, offA + i);
                }
                for (; i < len; i++)
                {
                    bufA[offA + i] -= bufB[offB + i];
                }
                return a;
            }

            var bBroad = b.BroadcastTo(a.Shape);
            bBroad.ForEachCoordinate(coords => a[coords] -= bBroad[coords]);
            return a;
        }

        /// <summary>
        /// Multiplies tensor a by tensor b in-place: a *= b (double precision).
        /// </summary>
        public static Tensor<double> Multiply_(this Tensor<double> a, Tensor<double> b)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));

            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int len = a.Length;
                int vecSize = Vector<double>.Count;
                int i = 0;

                for (; i <= len - vecSize; i += vecSize)
                {
                    var va = new Vector<double>(bufA, offA + i);
                    var vb = new Vector<double>(bufB, offB + i);
                    (va * vb).CopyTo(bufA, offA + i);
                }
                for (; i < len; i++)
                {
                    bufA[offA + i] *= bufB[offB + i];
                }
                return a;
            }

            var bBroad = b.BroadcastTo(a.Shape);
            bBroad.ForEachCoordinate(coords => a[coords] *= bBroad[coords]);
            return a;
        }

        /// <summary>
        /// Divides tensor a by tensor b element-wise in-place: a /= b (double precision).
        /// </summary>
        public static Tensor<double> Divide_(this Tensor<double> a, Tensor<double> b)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));

            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var bufA = a.Buffer;
                var bufB = b.Buffer;
                int offA = a.Offset;
                int offB = b.Offset;
                int len = a.Length;
                int vecSize = Vector<double>.Count;
                int i = 0;

                for (; i <= len - vecSize; i += vecSize)
                {
                    var va = new Vector<double>(bufA, offA + i);
                    var vb = new Vector<double>(bufB, offB + i);
                    (va / vb).CopyTo(bufA, offA + i);
                }
                for (; i < len; i++)
                {
                    bufA[offA + i] /= bufB[offB + i];
                }
                return a;
            }

            var bBroad = b.BroadcastTo(a.Shape);
            bBroad.ForEachCoordinate(coords => a[coords] /= bBroad[coords]);
            return a;
        }

        /// <summary>
        /// Adds a scalar to tensor a in-place: a += scalar (double precision).
        /// </summary>
        public static Tensor<double> AddScalar_(this Tensor<double> a, double scalar)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (a.IsContiguous)
            {
                var bufA = a.Buffer;
                int offA = a.Offset;
                int len = a.Length;
                var vScalar = new Vector<double>(scalar);
                int vecSize = Vector<double>.Count;
                int i = 0;
                for (; i <= len - vecSize; i += vecSize)
                {
                    var v = new Vector<double>(bufA, offA + i);
                    (v + vScalar).CopyTo(bufA, offA + i);
                }
                for (; i < len; i++) bufA[offA + i] += scalar;
                return a;
            }

            a.ForEachCoordinate(coords => a[coords] += scalar);
            return a;
        }

        /// <summary>
        /// Multiplies tensor a by a scalar in-place: a *= scalar (double precision).
        /// </summary>
        public static Tensor<double> MultiplyScalar_(this Tensor<double> a, double scalar)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (a.IsContiguous)
            {
                var bufA = a.Buffer;
                int offA = a.Offset;
                int len = a.Length;
                var vScalar = new Vector<double>(scalar);
                int vecSize = Vector<double>.Count;
                int i = 0;
                for (; i <= len - vecSize; i += vecSize)
                {
                    var v = new Vector<double>(bufA, offA + i);
                    (v * vScalar).CopyTo(bufA, offA + i);
                }
                for (; i < len; i++) bufA[offA + i] *= scalar;
                return a;
            }

            a.ForEachCoordinate(coords => a[coords] *= scalar);
            return a;
        }

        /// <summary>
        /// Applies ReLU activation in-place: a = max(0, a) (double precision).
        /// </summary>
        public static Tensor<double> Relu_(this Tensor<double> a)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (a.IsContiguous)
            {
                var bufA = a.Buffer;
                int offA = a.Offset;
                int len = a.Length;
                var zeroVec = Vector<double>.Zero;
                int vecSize = Vector<double>.Count;
                int i = 0;
                for (; i <= len - vecSize; i += vecSize)
                {
                    var v = new Vector<double>(bufA, offA + i);
                    Vector.Max(v, zeroVec).CopyTo(bufA, offA + i);
                }
                for (; i < len; i++) bufA[offA + i] = Math.Max(0.0, bufA[offA + i]);
                return a;
            }

            a.ForEachCoordinate(coords => a[coords] = Math.Max(0.0, a[coords]));
            return a;
        }

        /// <summary>
        /// Clamps all elements in-place: a = clamp(a, min, max) (double precision).
        /// </summary>
        public static Tensor<double> Clamp_(this Tensor<double> a, double min, double max)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (a.IsContiguous)
            {
                var bufA = a.Buffer;
                int offA = a.Offset;
                int len = a.Length;
                for (int i = 0; i < len; i++)
                {
                    double val = bufA[offA + i];
                    bufA[offA + i] = val < min ? min : (val > max ? max : val);
                }
                return a;
            }

            a.ForEachCoordinate(coords =>
            {
                double val = a[coords];
                a[coords] = val < min ? min : (val > max ? max : val);
            });
            return a;
        }

        /// <summary>
        /// Computes destination = a + b without allocating a new tensor buffer (double precision).
        /// </summary>
        public static void Add(Tensor<double> a, Tensor<double> b, Tensor<double> destination)
        {
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (a.Shape == b.Shape && a.Shape == destination.Shape && a.IsContiguous && b.IsContiguous && destination.IsContiguous)
            {
                var bufA = a.Buffer; int offA = a.Offset;
                var bufB = b.Buffer; int offB = b.Offset;
                var bufDst = destination.Buffer; int offDst = destination.Offset;
                int len = a.Length;
                int vecSize = Vector<double>.Count;
                int i = 0;

                for (; i <= len - vecSize; i += vecSize)
                {
                    var va = new Vector<double>(bufA, offA + i);
                    var vb = new Vector<double>(bufB, offB + i);
                    (va + vb).CopyTo(bufDst, offDst + i);
                }
                for (; i < len; i++)
                {
                    bufDst[offDst + i] = bufA[offA + i] + bufB[offB + i];
                }
                return;
            }

            var res = a + b;
            res.CopyTo(destination);
        }

        /// <summary>
        /// Computes destination = a - b without allocating a new tensor buffer (double precision).
        /// </summary>
        public static void Subtract(Tensor<double> a, Tensor<double> b, Tensor<double> destination)
        {
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (a.Shape == b.Shape && a.Shape == destination.Shape && a.IsContiguous && b.IsContiguous && destination.IsContiguous)
            {
                var bufA = a.Buffer; int offA = a.Offset;
                var bufB = b.Buffer; int offB = b.Offset;
                var bufDst = destination.Buffer; int offDst = destination.Offset;
                int len = a.Length;
                int vecSize = Vector<double>.Count;
                int i = 0;

                for (; i <= len - vecSize; i += vecSize)
                {
                    var va = new Vector<double>(bufA, offA + i);
                    var vb = new Vector<double>(bufB, offB + i);
                    (va - vb).CopyTo(bufDst, offDst + i);
                }
                for (; i < len; i++)
                {
                    bufDst[offDst + i] = bufA[offA + i] - bufB[offB + i];
                }
                return;
            }

            var res = a - b;
            res.CopyTo(destination);
        }

        /// <summary>
        /// Computes destination = a * b without allocating a new tensor buffer (double precision).
        /// </summary>
        public static void Multiply(Tensor<double> a, Tensor<double> b, Tensor<double> destination)
        {
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (a.Shape == b.Shape && a.Shape == destination.Shape && a.IsContiguous && b.IsContiguous && destination.IsContiguous)
            {
                var bufA = a.Buffer; int offA = a.Offset;
                var bufB = b.Buffer; int offB = b.Offset;
                var bufDst = destination.Buffer; int offDst = destination.Offset;
                int len = a.Length;
                int vecSize = Vector<double>.Count;
                int i = 0;

                for (; i <= len - vecSize; i += vecSize)
                {
                    var va = new Vector<double>(bufA, offA + i);
                    var vb = new Vector<double>(bufB, offB + i);
                    (va * vb).CopyTo(bufDst, offDst + i);
                }
                for (; i < len; i++)
                {
                    bufDst[offDst + i] = bufA[offA + i] * bufB[offB + i];
                }
                return;
            }

            var res = a * b;
            res.CopyTo(destination);
        }

        #endregion
    }
}
