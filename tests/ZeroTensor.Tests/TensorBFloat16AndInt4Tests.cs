using System;
using Xunit;
using ZeroTensor.Core;

namespace ZeroTensor.Tests
{
    public class TensorBFloat16AndInt4Tests
    {
        [Fact]
        public void BFloat16_ConversionAndArithmetic()
        {
            float orig = 3.14159f;
            BFloat16 bf = (BFloat16)orig;
            float convertedBack = (float)bf;

            // BFloat16 has 7 mantissa bits (~2-3 decimal digits accuracy)
            Assert.Equal(orig, convertedBack, precision: 2);

            var a = new Tensor<BFloat16>(2, 2);
            var b = new Tensor<BFloat16>(2, 2);

            a[0, 0] = (BFloat16)2.0f; a[0, 1] = (BFloat16)4.0f;
            a[1, 0] = (BFloat16)6.0f; a[1, 1] = (BFloat16)8.0f;

            b[0, 0] = (BFloat16)1.0f; b[0, 1] = (BFloat16)2.0f;
            b[1, 0] = (BFloat16)3.0f; b[1, 1] = (BFloat16)4.0f;

            var sum = TensorOps.Add(a, b);
            Assert.Equal((BFloat16)3.0f, sum[0, 0]);
            Assert.Equal((BFloat16)6.0f, sum[0, 1]);
            Assert.Equal((BFloat16)9.0f, sum[1, 0]);
            Assert.Equal((BFloat16)12.0f, sum[1, 1]);

            var diff = TensorOps.Subtract(a, b);
            Assert.Equal((BFloat16)1.0f, diff[0, 0]);
            Assert.Equal((BFloat16)2.0f, diff[0, 1]);

            var prod = TensorOps.Multiply(a, b);
            Assert.Equal((BFloat16)2.0f, prod[0, 0]);
            Assert.Equal((BFloat16)8.0f, prod[0, 1]);

            var div = TensorOps.Divide(a, b);
            Assert.Equal((BFloat16)2.0f, div[0, 0]);
            Assert.Equal((BFloat16)2.0f, div[0, 1]);
        }

        [Fact]
        public void BFloat16_MatMul_CacheTiled()
        {
            // A: (2 x 2), B: (2 x 2)
            var a = new Tensor<BFloat16>(2, 2);
            a[0, 0] = (BFloat16)1f; a[0, 1] = (BFloat16)2f;
            a[1, 0] = (BFloat16)3f; a[1, 1] = (BFloat16)4f;

            var b = new Tensor<BFloat16>(2, 2);
            b[0, 0] = (BFloat16)5f; b[0, 1] = (BFloat16)6f;
            b[1, 0] = (BFloat16)7f; b[1, 1] = (BFloat16)8f;

            // C = A @ B
            // C[0,0] = 1*5 + 2*7 = 19
            // C[0,1] = 1*6 + 2*8 = 22
            // C[1,0] = 3*5 + 4*7 = 43
            // C[1,1] = 3*6 + 4*8 = 50
            var c = TensorBlas.MatMul(a, b);

            Assert.Equal((BFloat16)19f, c[0, 0]);
            Assert.Equal((BFloat16)22f, c[0, 1]);
            Assert.Equal((BFloat16)43f, c[1, 0]);
            Assert.Equal((BFloat16)50f, c[1, 1]);
        }

        [Fact]
        public void GemmInt4_PackedMultiplication()
        {
            // Activation: (1 x 4)
            var act = new Tensor<float>(1, 4);
            act[0, 0] = 1.0f;
            act[0, 1] = 2.0f;
            act[0, 2] = 3.0f;
            act[0, 3] = 4.0f;

            // Packed weights: (2 x 1) -> 2 bytes representing 4 weights for 1 column
            // Byte 0: low nibble = 2, high nibble = 3 (weights at row 0, 1)
            // Byte 1: low nibble = 4, high nibble = 1 (weights at row 2, 3)
            var packed = new Tensor<byte>(2, 1);
            packed[0, 0] = (byte)(2 | (3 << 4)); // 0x32
            packed[1, 0] = (byte)(4 | (1 << 4)); // 0x14

            var scales = new Tensor<float>(1);
            scales[0] = 0.5f;

            var zp = new Tensor<float>(1);
            zp[0] = 0.0f;

            // Expected weights:
            // w[0] = 2 * 0.5 = 1.0
            // w[1] = 3 * 0.5 = 1.5
            // w[2] = 4 * 0.5 = 2.0
            // w[3] = 1 * 0.5 = 0.5
            // Result = 1.0*1.0 + 2.0*1.5 + 3.0*2.0 + 4.0*0.5 = 1.0 + 3.0 + 6.0 + 2.0 = 12.0
            var res = TensorBlas.GemmInt4(act, packed, scales, zp);

            Assert.Equal(1, res.Shape[0]);
            Assert.Equal(1, res.Shape[1]);
            Assert.Equal(12.0f, res[0, 0], precision: 3);
        }
    }
}
