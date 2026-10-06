using System;
using Xunit;
using ZeroTensor.Core;

namespace ZeroTensor.Tests
{
    public class NeuralOperatorsTests
    {
        [Fact]
        public void SoftmaxFast_NumericalStabilityAndSumToOne()
        {
            // Large values that would overflow naive exp without max subtraction
            var t = Tensor.FromArray(new float[] { 1000f, 1001f, 1002f }, 1, 3);
            var sm = TensorOps.SoftmaxFast(t, axis: -1);

            Assert.False(float.IsNaN(sm[0, 0]));
            Assert.False(float.IsNaN(sm[0, 1]));
            Assert.False(float.IsNaN(sm[0, 2]));

            float sum = sm[0, 0] + sm[0, 1] + sm[0, 2];
            Assert.True(Math.Abs(sum - 1.0f) < 1e-5f);

            // sm[0, 2] should be the largest
            Assert.True(sm[0, 2] > sm[0, 1]);
            Assert.True(sm[0, 1] > sm[0, 0]);
        }

        [Fact]
        public void RMSNorm_Correctness()
        {
            // x = [1, 2, 3, 4], d = 4
            // sum(x^2) = 1 + 4 + 9 + 16 = 30
            // mean(x^2) = 30 / 4 = 7.5
            // rms = sqrt(7.5 + 1e-5) = 2.738613
            var x = Tensor.FromArray(new float[] { 1f, 2f, 3f, 4f }, 1, 4);
            var weight = Tensor.FromArray(new float[] { 2f, 2f, 2f, 2f }, 4);

            var y = TensorOps.RMSNorm(x, weight, eps: 1e-5f);

            float expectedRms = (float)Math.Sqrt(7.5 + 1e-5);
            for (int i = 0; i < 4; i++)
            {
                float expectedVal = (x[0, i] / expectedRms) * 2f;
                Assert.True(Math.Abs(y[0, i] - expectedVal) < 1e-4f, $"Mismatch at index {i}: actual={y[0, i]}, expected={expectedVal}");
            }
        }

        [Fact]
        public void LayerNorm_Correctness()
        {
            // x = [2, 4, 6, 8], mean = 5
            // diff = [-3, -1, 1, 3]
            // var = (9 + 1 + 1 + 9) / 4 = 5
            // std = sqrt(5 + 1e-5) = 2.23607
            var x = Tensor.FromArray(new float[] { 2f, 4f, 6f, 8f }, 1, 4);
            var weight = Tensor.FromArray(new float[] { 1f, 1f, 1f, 1f }, 4);
            var bias = Tensor.FromArray(new float[] { 0.1f, 0.2f, 0.3f, 0.4f }, 4);

            var y = TensorOps.LayerNorm(x, weight, bias, eps: 1e-5f);

            float expectedStd = (float)Math.Sqrt(5.0 + 1e-5);
            float[] diffs = { -3f, -1f, 1f, 3f };
            float[] biases = { 0.1f, 0.2f, 0.3f, 0.4f };

            for (int i = 0; i < 4; i++)
            {
                float expectedVal = (diffs[i] / expectedStd) * 1f + biases[i];
                Assert.True(Math.Abs(y[0, i] - expectedVal) < 1e-4f, $"Mismatch at {i}: actual={y[0, i]}, expected={expectedVal}");
            }
        }

        [Fact]
        public void SiLU_Correctness()
        {
            var x = Tensor.FromArray(new float[] { -2f, 0f, 2f }, 3);
            var y = TensorOps.SiLU(x);

            // SiLU(0) = 0 * 0.5 = 0
            Assert.Equal(0f, y[1]);

            // SiLU(2) = 2 / (1 + exp(-2)) = 2 / (1 + 0.135335) = 1.76159
            float expected2 = 2.0f / (1.0f + (float)Math.Exp(-2.0));
            Assert.True(Math.Abs(y[2] - expected2) < 1e-5f);

            // SiLU(-2) = -2 / (1 + exp(2)) = -2 / (1 + 7.38905) = -0.2384
            float expectedNeg2 = -2.0f / (1.0f + (float)Math.Exp(2.0));
            Assert.True(Math.Abs(y[0] - expectedNeg2) < 1e-5f);
        }

        [Fact]
        public void GELU_Correctness()
        {
            var x = Tensor.FromArray(new float[] { 0f, 1f, -1f }, 3);
            var y = TensorOps.GELU(x);

            // GELU(0) = 0
            Assert.Equal(0f, y[0]);

            // GELU(1) ~= 0.8413
            Assert.True(Math.Abs(y[1] - 0.8413f) < 1e-3f);

            // GELU(-1) ~= -0.1587
            Assert.True(Math.Abs(y[2] - (-0.1587f)) < 1e-3f);
        }

        [Fact]
        public void ScaledDotProductAttention_Batched()
        {
            // Batch=1, Heads=2, Seq=3, Dim=4
            var q = Tensor.Zeros<float>(1, 2, 3, 4);
            var k = Tensor.Zeros<float>(1, 2, 3, 4);
            var v = Tensor.Zeros<float>(1, 2, 3, 4);

            q.Fill(1f);
            k.Fill(1f);
            v.Fill(2f);

            var outAttn = TensorOps.ScaledDotProductAttention(q, k, v);

            Assert.Equal(new[] { 1, 2, 3, 4 }, outAttn.Shape.Dimensions);

            // Since all Q and K are identical, attention weights are uniform (1/3 each),
            // and V is all 2.0, so output should be all 2.0
            for (int h = 0; h < 2; h++)
            {
                for (int s = 0; s < 3; s++)
                {
                    for (int d = 0; d < 4; d++)
                    {
                        Assert.True(Math.Abs(outAttn[0, h, s, d] - 2f) < 1e-4f);
                    }
                }
            }
        }

        [Fact]
        public void Conv2D_IdentityFilter()
        {
            // Input: 1 batch, 1 channel, 3x3 image
            var input = Tensor.FromArray(new float[]
            {
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f
            }, 1, 1, 3, 3);

            // Weight: 1 out channel, 1 in channel, 1x1 filter with value 1 (identity)
            var weight = Tensor.FromArray(new float[] { 1f }, 1, 1, 1, 1);
            var bias = Tensor.FromArray(new float[] { 0.5f }, 1);

            var output = TensorOps.Conv2D(input, weight, bias, stride: 1, padding: 0);

            Assert.Equal(new[] { 1, 1, 3, 3 }, output.Shape.Dimensions);

            // Output should be input + 0.5
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++)
                {
                    Assert.Equal(input[0, 0, r, c] + 0.5f, output[0, 0, r, c]);
                }
            }
        }
    }
}
