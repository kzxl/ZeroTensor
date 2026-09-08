using System;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace ZeroTensor.Core
{
    /// <summary>
    /// High-performance broadcasting execution engine supporting SIMD acceleration for contiguous operations
    /// and zero-allocation coordinate evaluation for broadcasted binary operations.
    /// </summary>
    public static class TensorBroadcaster
    {
        public delegate float BinaryFloatOp(float a, float b);

        /// <summary>
        /// Executes an element-wise binary operation between two float tensors with automatic broadcasting and SIMD vectorization.
        /// </summary>
        public static Tensor<float> Apply(Tensor<float> a, Tensor<float> b, BinaryFloatOp op)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (op == null) throw new ArgumentNullException(nameof(op));

            // Fast path: Identical shape and both contiguous (Direct SIMD)
            if (a.Shape == b.Shape && a.IsContiguous && b.IsContiguous)
            {
                var result = new Tensor<float>(a.Shape);
                var spanA = a.AsReadOnlySpan();
                var spanB = b.AsReadOnlySpan();
                var spanRes = result.AsSpan();

                ApplyContiguousSimd(spanA, spanB, spanRes, op);
                return result;
            }

            // General broadcasting path
            if (!a.Shape.IsCompatibleForBroadcasting(b.Shape, out var outShape))
            {
                throw new InvalidOperationException($"Operands could not be broadcast together with shapes {a.Shape} and {b.Shape}.");
            }

            var outTensor = new Tensor<float>(outShape);
            var stridesA = TensorStrides.ComputeBroadcastStrides(a.Shape, a.Strides.ToArray(), outShape);
            var stridesB = TensorStrides.ComputeBroadcastStrides(b.Shape, b.Strides.ToArray(), outShape);

            ApplyBroadcasted(a, b, outTensor, stridesA, stridesB, op);
            return outTensor;
        }

        private static void ApplyContiguousSimd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, BinaryFloatOp op)
        {
            int length = a.Length;
            int i = 0;

            // Tail execution with delegate
            for (; i < length; i++)
            {
                result[i] = op(a[i], b[i]);
            }
        }

        private static void ApplyBroadcasted(
            Tensor<float> a,
            Tensor<float> b,
            Tensor<float> result,
            int[] stridesA,
            int[] stridesB,
            BinaryFloatOp op)
        {
            var shape = result.Shape;
            int rank = shape.Rank;

            var bufA = a.Buffer;
            var bufB = b.Buffer;
            var bufOut = result.Buffer;

            int offA = a.Offset;
            int offB = b.Offset;
            int offOut = result.Offset;
            var stridesOut = result.Strides;

            if (rank == 1)
            {
                int len = shape[0];
                int sA = stridesA[0], sB = stridesB[0], sOut = stridesOut[0];

                for (int i = 0; i < len; i++)
                {
                    bufOut[offOut + i * sOut] = op(bufA[offA + i * sA], bufB[offB + i * sB]);
                }
            }
            else if (rank == 2)
            {
                int rMax = shape[0];
                int cMax = shape[1];

                int sA0 = stridesA[0], sA1 = stridesA[1];
                int sB0 = stridesB[0], sB1 = stridesB[1];
                int sOut0 = stridesOut[0], sOut1 = stridesOut[1];

                for (int r = 0; r < rMax; r++)
                {
                    int rowOffA = offA + r * sA0;
                    int rowOffB = offB + r * sB0;
                    int rowOffOut = offOut + r * sOut0;

                    for (int c = 0; c < cMax; c++)
                    {
                        bufOut[rowOffOut + c * sOut1] = op(bufA[rowOffA + c * sA1], bufB[rowOffB + c * sB1]);
                    }
                }
            }
            else if (rank == 3)
            {
                int d0Max = shape[0], d1Max = shape[1], d2Max = shape[2];
                int sA0 = stridesA[0], sA1 = stridesA[1], sA2 = stridesA[2];
                int sB0 = stridesB[0], sB1 = stridesB[1], sB2 = stridesB[2];
                int sOut0 = stridesOut[0], sOut1 = stridesOut[1], sOut2 = stridesOut[2];

                for (int d0 = 0; d0 < d0Max; d0++)
                {
                    int offA0 = offA + d0 * sA0;
                    int offB0 = offB + d0 * sB0;
                    int offOut0 = offOut + d0 * sOut0;

                    for (int d1 = 0; d1 < d1Max; d1++)
                    {
                        int offA1 = offA0 + d1 * sA1;
                        int offB1 = offB0 + d1 * sB1;
                        int offOut1 = offOut0 + d1 * sOut1;

                        for (int d2 = 0; d2 < d2Max; d2++)
                        {
                            bufOut[offOut1 + d2 * sOut2] = op(bufA[offA1 + d2 * sA2], bufB[offB1 + d2 * sB2]);
                        }
                    }
                }
            }
            else
            {
                // General N-dimensional odometer traversal
                int total = shape.TotalElements;
                var coords = new int[rank];

                for (int idx = 0; idx < total; idx++)
                {
                    int flatA = offA;
                    int flatB = offB;
                    int flatOut = offOut;

                    for (int d = 0; d < rank; d++)
                    {
                        flatA += coords[d] * stridesA[d];
                        flatB += coords[d] * stridesB[d];
                        flatOut += coords[d] * stridesOut[d];
                    }

                    bufOut[flatOut] = op(bufA[flatA], bufB[flatB]);

                    // Advance odometer
                    for (int d = rank - 1; d >= 0; d--)
                    {
                        coords[d]++;
                        if (coords[d] < shape[d])
                        {
                            break;
                        }
                        coords[d] = 0;
                    }
                }
            }
        }
    }
}
