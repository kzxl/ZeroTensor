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

        [Fact]
        public void EmbeddingLookup_Correctness()
        {
            // Vocab = 4, Dim = 3
            var weight = Tensor.FromArray(new float[]
            {
                10f, 11f, 12f, // token 0
                20f, 21f, 22f, // token 1
                30f, 31f, 32f, // token 2
                40f, 41f, 42f  // token 3
            }, 4, 3);

            // Batch = 2, Seq = 2
            var indices = Tensor.FromArray(new int[]
            {
                3, 0,
                1, 2
            }, 2, 2);

            var emb = TensorOps.EmbeddingLookup(indices, weight);

            Assert.Equal(new[] { 2, 2, 3 }, emb.Shape.Dimensions);

            // batch 0, seq 0 -> token 3
            Assert.Equal(40f, emb[0, 0, 0]);
            Assert.Equal(41f, emb[0, 0, 1]);
            Assert.Equal(42f, emb[0, 0, 2]);

            // batch 0, seq 1 -> token 0
            Assert.Equal(10f, emb[0, 1, 0]);

            // batch 1, seq 0 -> token 1
            Assert.Equal(20f, emb[1, 0, 0]);

            // batch 1, seq 1 -> token 2
            Assert.Equal(30f, emb[1, 1, 0]);
        }

        [Fact]
        public void ApplyRoPE_NormPreservation()
        {
            // Seq = 2, HeadDim = 4
            var x = Tensor.FromArray(new float[]
            {
                1f, 2f, 3f, 4f, // pos 0
                5f, 6f, 7f, 8f  // pos 1
            }, 1, 1, 2, 4);

            var xRot = TensorOps.ApplyRoPE(x, startPos: 0);

            Assert.Equal(new[] { 1, 1, 2, 4 }, xRot.Shape.Dimensions);

            // At pos = 0, angle = 0 -> cos = 1, sin = 0, so vector must be unchanged!
            Assert.Equal(1f, xRot[0, 0, 0, 0]);
            Assert.Equal(2f, xRot[0, 0, 0, 1]);
            Assert.Equal(3f, xRot[0, 0, 0, 2]);
            Assert.Equal(4f, xRot[0, 0, 0, 3]);

            // At pos = 1, rotation is orthogonal, so L2 norm squared must be preserved!
            float normOrigSq = 5f * 5f + 6f * 6f + 7f * 7f + 8f * 8f; // 25 + 36 + 49 + 64 = 174
            float r0 = xRot[0, 0, 1, 0];
            float r1 = xRot[0, 0, 1, 1];
            float r2 = xRot[0, 0, 1, 2];
            float r3 = xRot[0, 0, 1, 3];
            float normRotSq = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3;

            Assert.True(Math.Abs(normOrigSq - normRotSq) < 1e-3f, $"Norm not preserved: orig={normOrigSq}, rot={normRotSq}");
        }

        [Fact]
        public void KVCache_AppendAndRetrieveValid()
        {
            // Batch = 1, Heads = 2, MaxSeq = 10, Dim = 4
            using var cache = new ZeroTensor.Core.Neural.KVCache<float>(1, 2, 10, 4);
            Assert.Equal(0, cache.CurrentLength);

            // 1. Prefill 3 tokens
            var kPrefill = Tensor.Ones(1, 2, 3, 4);
            var vPrefill = Tensor.Full(2f, 1, 2, 3, 4);
            cache.Append(kPrefill, vPrefill);
            Assert.Equal(3, cache.CurrentLength);

            var validK = cache.GetValidKeys();
            var validV = cache.GetValidValues();
            Assert.Equal(new[] { 1, 2, 3, 4 }, validK.Shape.Dimensions);
            Assert.Equal(new[] { 1, 2, 3, 4 }, validV.Shape.Dimensions);
            Assert.Equal(1f, validK[0, 0, 2, 0]);
            Assert.Equal(2f, validV[0, 0, 2, 0]);

            // 2. Decode 1 new token
            var kNext = Tensor.Full(3f, 1, 2, 1, 4);
            var vNext = Tensor.Full(4f, 1, 2, 1, 4);
            cache.Append(kNext, vNext);
            Assert.Equal(4, cache.CurrentLength);

            validK = cache.GetValidKeys();
            validV = cache.GetValidValues();
            Assert.Equal(new[] { 1, 2, 4, 4 }, validK.Shape.Dimensions);
            Assert.Equal(3f, validK[0, 0, 3, 0]);
            Assert.Equal(4f, validV[0, 0, 3, 0]);

            // 3. Reset
            cache.Reset();
            Assert.Equal(0, cache.CurrentLength);
        }

        [Fact]
        public void MaxPool2D_2x2_Stride2()
        {
            var input = Tensor.FromArray(new float[]
            {
                1f, 3f, 2f, 4f,
                5f, 6f, 7f, 8f,
                9f, 2f, 3f, 1f,
                4f, 8f, 5f, 6f
            }, 1, 1, 4, 4);

            var pooled = TensorOps.MaxPool2D(input, kernelSize: 2, stride: 2);

            Assert.Equal(new[] { 1, 1, 2, 2 }, pooled.Shape.Dimensions);
            // Window (0,0): max(1, 3, 5, 6) = 6
            Assert.Equal(6f, pooled[0, 0, 0, 0]);
            // Window (0,1): max(2, 4, 7, 8) = 8
            Assert.Equal(8f, pooled[0, 0, 0, 1]);
            // Window (1,0): max(9, 2, 4, 8) = 9
            Assert.Equal(9f, pooled[0, 0, 1, 0]);
            // Window (1,1): max(3, 1, 5, 6) = 6
            Assert.Equal(6f, pooled[0, 0, 1, 1]);
        }

        [Fact]
        public void AvgPool2D_2x2_Stride2()
        {
            var input = Tensor.FromArray(new float[]
            {
                1f, 3f, 2f, 4f,
                5f, 7f, 6f, 8f,
                1f, 1f, 2f, 2f,
                3f, 3f, 4f, 4f
            }, 1, 1, 4, 4);

            var pooled = TensorOps.AvgPool2D(input, kernelSize: 2, stride: 2);

            Assert.Equal(new[] { 1, 1, 2, 2 }, pooled.Shape.Dimensions);
            // Window (0,0): (1 + 3 + 5 + 7) / 4 = 16 / 4 = 4
            Assert.Equal(4f, pooled[0, 0, 0, 0]);
            // Window (0,1): (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5
            Assert.Equal(5f, pooled[0, 0, 0, 1]);
            // Window (1,0): (1 + 1 + 3 + 3) / 4 = 8 / 4 = 2
            Assert.Equal(2f, pooled[0, 0, 1, 0]);
            // Window (1,1): (2 + 2 + 4 + 4) / 4 = 12 / 4 = 3
            Assert.Equal(3f, pooled[0, 0, 1, 1]);
        }

        [Fact]
        public void Interpolate2D_NearestAndBilinear()
        {
            // 2x2 image
            var input = Tensor.FromArray(new float[]
            {
                10f, 20f,
                30f, 40f
            }, 1, 1, 2, 2);

            // 1. Nearest upsample to 4x4
            var nearest = TensorOps.Interpolate2D(input, 4, 4, InterpolationMode.Nearest);
            Assert.Equal(new[] { 1, 1, 4, 4 }, nearest.Shape.Dimensions);
            Assert.Equal(10f, nearest[0, 0, 0, 0]);
            Assert.Equal(10f, nearest[0, 0, 1, 1]);
            Assert.Equal(20f, nearest[0, 0, 0, 3]);
            Assert.Equal(40f, nearest[0, 0, 3, 3]);

            // 2. Bilinear upsample to 3x3 with alignCorners = true
            var bilinear = TensorOps.Interpolate2D(input, 3, 3, InterpolationMode.Bilinear, alignCorners: true);
            Assert.Equal(new[] { 1, 1, 3, 3 }, bilinear.Shape.Dimensions);
            // Center pixel must be average of all 4 corners: (10 + 20 + 30 + 40) / 4 = 25
            Assert.Equal(10f, bilinear[0, 0, 0, 0]);
            Assert.Equal(20f, bilinear[0, 0, 0, 2]);
            Assert.Equal(30f, bilinear[0, 0, 2, 0]);
            Assert.Equal(40f, bilinear[0, 0, 2, 2]);
            Assert.True(Math.Abs(bilinear[0, 0, 1, 1] - 25f) < 1e-4f);
        }

        [Fact]
        public void GroupNorm_4Channels_2Groups()
        {
            // 4 channels, 2 groups -> 2 channels per group
            // Spatial 2x2: each group has 2 * 2 * 2 = 8 values
            var input = Tensor.Zeros<float>(1, 4, 2, 2);
            // Group 0: channels 0, 1
            for (int c = 0; c < 2; c++)
                for (int r = 0; r < 2; r++)
                    for (int col = 0; col < 2; col++)
                        input[0, c, r, col] = (c * 4 + r * 2 + col); // values 0..7

            // Group 1: channels 2, 3
            for (int c = 2; c < 4; c++)
                for (int r = 0; r < 2; r++)
                    for (int col = 0; col < 2; col++)
                        input[0, c, r, col] = (c * 4 + r * 2 + col) * 10f;

            var gn = TensorOps.GroupNorm(input, numGroups: 2, eps: 1e-5f);
            Assert.Equal(new[] { 1, 4, 2, 2 }, gn.Shape.Dimensions);

            // Group 0 mean = 3.5, variance = 5.25, std = sqrt(5.25) = 2.291288
            // Check normalized mean across group 0 is approx 0
            float g0Sum = 0f;
            for (int c = 0; c < 2; c++)
                for (int r = 0; r < 2; r++)
                    for (int col = 0; col < 2; col++)
                        g0Sum += gn[0, c, r, col];

            Assert.True(Math.Abs(g0Sum) < 1e-4f);

            // Check normalized mean across group 1 is approx 0
            float g1Sum = 0f;
            for (int c = 2; c < 4; c++)
                for (int r = 0; r < 2; r++)
                    for (int col = 0; col < 2; col++)
                        g1Sum += gn[0, c, r, col];

            Assert.True(Math.Abs(g1Sum) < 1e-4f);
        }

        [Fact]
        public void FlashAttentionCpu_MatchesStandardAttention()
        {
            // Batch = 1, Heads = 2, Seq = 8, Dim = 16
            var q = Tensor.Zeros<float>(1, 2, 8, 16);
            var k = Tensor.Zeros<float>(1, 2, 8, 16);
            var v = Tensor.Zeros<float>(1, 2, 8, 16);

            for (int i = 0; i < 8; i++)
            {
                for (int d = 0; d < 16; d++)
                {
                    q[0, 0, i, d] = (i + d) * 0.1f;
                    k[0, 0, i, d] = (i - d) * 0.05f;
                    v[0, 0, i, d] = (i * 2 + d) * 0.1f;

                    q[0, 1, i, d] = (i - d) * 0.1f;
                    k[0, 1, i, d] = (i + d) * 0.05f;
                    v[0, 1, i, d] = (d - i) * 0.1f;
                }
            }

            var expected = TensorOps.ScaledDotProductAttention(q, k, v);
            var actual = TensorOps.FlashAttentionCpu(q, k, v);

            Assert.Equal(expected.Shape.Dimensions, actual.Shape.Dimensions);

            for (int h = 0; h < 2; h++)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int d = 0; d < 16; d++)
                    {
                        float diff = Math.Abs(expected[0, h, i, d] - actual[0, h, i, d]);
                        Assert.True(diff < 1e-4f, $"Mismatch at h={h}, i={i}, d={d}: expected {expected[0, h, i, d]}, actual {actual[0, h, i, d]}");
                    }
                }
            }
        }

        [Fact]
        public void Conv1D_BasicExecution()
        {
            // Input: [Batch=1, InC=1, L=5]
            var input = Tensor.FromArray(new float[] { 1f, 2f, 3f, 4f, 5f }, 1, 1, 5);
            // Weight: [OutC=1, InC=1, K=3]
            var weight = Tensor.FromArray(new float[] { 1f, 0f, -1f }, 1, 1, 3);
            var bias = Tensor.FromArray(new float[] { 10f }, 1);

            // Output length = (5 - 3) / 1 + 1 = 3
            // ol=0: 1*1 + 2*0 + 3*(-1) = -2 + 10 = 8
            // ol=1: 2*1 + 3*0 + 4*(-1) = -2 + 10 = 8
            // ol=2: 3*1 + 4*0 + 5*(-1) = -2 + 10 = 8
            var output = TensorOps.Conv1D(input, weight, bias, stride: 1, padding: 0);

            Assert.Equal(new[] { 1, 1, 3 }, output.Shape.Dimensions);
            Assert.Equal(8f, output[0, 0, 0]);
            Assert.Equal(8f, output[0, 0, 1]);
            Assert.Equal(8f, output[0, 0, 2]);
        }

        [Fact]
        public void ConvTranspose2D_BasicExecution()
        {
            // Input: [1, 1, 2, 2]
            var input = Tensor.FromArray(new float[]
            {
                1f, 2f,
                3f, 4f
            }, 1, 1, 2, 2);

            // Weight: [1, 1, 2, 2]
            var weight = Tensor.FromArray(new float[]
            {
                1f, 1f,
                1f, 1f
            }, 1, 1, 2, 2);

            // With stride = 2, output size is (2 - 1)*2 + 2 = 4x4
            var output = TensorOps.ConvTranspose2D(input, weight, stride: 2, padding: 0);

            Assert.Equal(new[] { 1, 1, 4, 4 }, output.Shape.Dimensions);

            // Top-left 2x2 should be input[0,0] * 1 = 1
            Assert.Equal(1f, output[0, 0, 0, 0]);
            Assert.Equal(1f, output[0, 0, 0, 1]);
            Assert.Equal(1f, output[0, 0, 1, 0]);
            Assert.Equal(1f, output[0, 0, 1, 1]);

            // Top-right 2x2 should be input[0,1] * 1 = 2
            Assert.Equal(2f, output[0, 0, 0, 2]);
            Assert.Equal(2f, output[0, 0, 0, 3]);
            Assert.Equal(2f, output[0, 0, 1, 2]);
            Assert.Equal(2f, output[0, 0, 1, 3]);

            // Bottom-right 2x2 should be input[1,1] * 1 = 4
            Assert.Equal(4f, output[0, 0, 2, 2]);
            Assert.Equal(4f, output[0, 0, 3, 3]);
        }

        [Fact]
        public void InferenceWorkspace_PingPongSwap()
        {
            using var ws = new InferenceWorkspace<float>(2, 4);

            var initialInput = Tensor.FromArray(new float[]
            {
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f
            }, 2, 4);

            ws.Reset(initialInput);

            Assert.Equal(1f, ws.CurrentInput[0, 0]);
            Assert.Equal(8f, ws.CurrentInput[1, 3]);

            // Simulate Layer 1 forward: CurrentOutput = CurrentInput * 2
            var inSpan = ws.CurrentInput.AsReadOnlySpan();
            var outSpan = ws.CurrentOutput.AsSpan();
            for (int i = 0; i < inSpan.Length; i++) outSpan[i] = inSpan[i] * 2f;

            // Step to Layer 2: buffers swap in O(1)
            ws.Step();

            Assert.Equal(2f, ws.CurrentInput[0, 0]);
            Assert.Equal(16f, ws.CurrentInput[1, 3]);

            // Simulate Layer 2 forward: CurrentOutput = CurrentInput + 10
            inSpan = ws.CurrentInput.AsReadOnlySpan();
            outSpan = ws.CurrentOutput.AsSpan();
            for (int i = 0; i < inSpan.Length; i++) outSpan[i] = inSpan[i] + 10f;

            // Step to Layer 3
            ws.Step();

            Assert.Equal(12f, ws.CurrentInput[0, 0]);
            Assert.Equal(26f, ws.CurrentInput[1, 3]);
        }
    }
}
