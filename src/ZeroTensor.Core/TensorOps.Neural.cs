using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace ZeroTensor.Core
{
    public static partial class TensorOps
    {
        #region Unsafe SIMD Vector Helpers

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector<float> ReadVector(ref float r, int offset) =>
            Unsafe.ReadUnaligned<Vector<float>>(ref Unsafe.As<float, byte>(ref Unsafe.Add(ref r, offset)));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteVector(ref float r, int offset, Vector<float> v) =>
            Unsafe.WriteUnaligned(ref Unsafe.As<float, byte>(ref Unsafe.Add(ref r, offset)), v);

        #endregion

        #region Softmax (Fused Single-Pass)

        /// <summary>
        /// Computes numerically stable Softmax along the specified axis.
        /// Uses high-performance fused SIMD kernel when operating on the last axis.
        /// </summary>
        public static Tensor<float> SoftmaxFast(Tensor<float> t, int axis = -1)
        {
            if (t == null) throw new ArgumentNullException(nameof(t));
            int ax = axis < 0 ? axis + t.Rank : axis;
            if (ax < 0 || ax >= t.Rank) throw new ArgumentOutOfRangeException(nameof(axis));

            // High-speed fused kernel along the contiguous last axis
            if (ax == t.Rank - 1 && t.IsContiguous)
            {
                var result = new Tensor<float>(t.Shape);
                int lastDim = t.Shape[ax];
                int batchRows = t.Length / lastDim;
                int vecSize = Vector<float>.Count;

                void ProcessRow(int r)
                {
                    int offset = r * lastDim;
                    ref float pSrc = ref t.Storage.GetPinnableReference(t.Offset + offset);
                    ref float pDst = ref result.Storage.GetPinnableReference(result.Offset + offset);

                    // 1. Find max for numerical stability
                    float maxVal = pSrc;
                    int i = 0;

                    if (Vector.IsHardwareAccelerated && lastDim >= vecSize)
                    {
                        var maxVec = new Vector<float>(maxVal);
                        for (; i <= lastDim - vecSize; i += vecSize)
                        {
                            var v = ReadVector(ref pSrc, i);
                            maxVec = Vector.Max(maxVec, v);
                        }
                        for (int k = 0; k < vecSize; k++)
                        {
                            if (maxVec[k] > maxVal) maxVal = maxVec[k];
                        }
                    }

                    for (; i < lastDim; i++)
                    {
                        float val = Unsafe.Add(ref pSrc, i);
                        if (val > maxVal) maxVal = val;
                    }

                    // 2. Compute exp(x - max) and accumulate sum
                    float sum = 0f;
                    for (int j = 0; j < lastDim; j++)
                    {
                        float val = Unsafe.Add(ref pSrc, j);
                        float e = (float)Math.Exp(val - maxVal);
                        Unsafe.Add(ref pDst, j) = e;
                        sum += e;
                    }

                    // 3. Normalize
                    float invSum = 1.0f / (sum > 0f ? sum : 1e-12f);
                    i = 0;
                    if (Vector.IsHardwareAccelerated && lastDim >= vecSize)
                    {
                        var invVec = new Vector<float>(invSum);
                        for (; i <= lastDim - vecSize; i += vecSize)
                        {
                            var v = ReadVector(ref pDst, i);
                            WriteVector(ref pDst, i, v * invVec);
                        }
                    }

                    for (; i < lastDim; i++)
                    {
                        Unsafe.Add(ref pDst, i) *= invSum;
                    }
                }

                if (batchRows > 1)
                {
                    Parallel.For(0, batchRows, ProcessRow);
                }
                else
                {
                    ProcessRow(0);
                }

                return result;
            }

            // General axis fallback using existing reduction
            return SoftmaxReduce(t, ax);
        }

        #endregion

        #region RMSNorm & LayerNorm

        /// <summary>
        /// Root Mean Square Normalization (RMSNorm) widely used in modern LLMs (LLaMA, Mistral, Gemma, Qwen).
        /// y = (x / sqrt(mean(x^2) + eps)) * weight
        /// </summary>
        public static Tensor<float> RMSNorm(Tensor<float> x, Tensor<float>? weight = null, float eps = 1e-5f)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            if (x.Rank < 1) throw new ArgumentException("RMSNorm requires rank >= 1.", nameof(x));

            int normDim = x.Shape[x.Rank - 1];
            if (weight != null && (weight.Length != normDim || weight.Rank != 1))
            {
                throw new ArgumentException($"RMSNorm weight must be 1D with length {normDim}.", nameof(weight));
            }

            var contigX = x.IsContiguous ? x : x.ToContiguous();
            var contigW = weight != null ? (weight.IsContiguous ? weight : weight.ToContiguous()) : null;
            var result = new Tensor<float>(contigX.Shape);
            int batchRows = contigX.Length / normDim;

            float invNormDim = 1.0f / normDim;
            int vecSize = Vector<float>.Count;

            void ProcessRow(int r)
            {
                int offset = r * normDim;
                ref float pSrc = ref contigX.Storage.GetPinnableReference(contigX.Offset + offset);
                ref float pDst = ref result.Storage.GetPinnableReference(result.Offset + offset);

                // 1. Calculate sum of squares
                float sumSq = 0f;
                int i = 0;

                if (Vector.IsHardwareAccelerated && normDim >= vecSize)
                {
                    var sumVec = Vector<float>.Zero;
                    for (; i <= normDim - vecSize; i += vecSize)
                    {
                        var v = ReadVector(ref pSrc, i);
                        sumVec += v * v;
                    }
                    for (int k = 0; k < vecSize; k++) sumSq += sumVec[k];
                }

                for (; i < normDim; i++)
                {
                    float val = Unsafe.Add(ref pSrc, i);
                    sumSq += val * val;
                }

                // 2. RMS factor = 1 / sqrt(mean(x^2) + eps)
                float rms = (float)(1.0 / Math.Sqrt(sumSq * invNormDim + eps));
                var rmsVec = new Vector<float>(rms);

                // 3. Normalize and scale
                i = 0;
                if (contigW != null)
                {
                    ref float pWeight = ref contigW.Storage.GetPinnableReference(contigW.Offset);
                    if (Vector.IsHardwareAccelerated && normDim >= vecSize)
                    {
                        for (; i <= normDim - vecSize; i += vecSize)
                        {
                            var v = ReadVector(ref pSrc, i);
                            var w = ReadVector(ref pWeight, i);
                            WriteVector(ref pDst, i, v * rmsVec * w);
                        }
                    }

                    for (; i < normDim; i++)
                    {
                        Unsafe.Add(ref pDst, i) = Unsafe.Add(ref pSrc, i) * rms * Unsafe.Add(ref pWeight, i);
                    }
                }
                else
                {
                    if (Vector.IsHardwareAccelerated && normDim >= vecSize)
                    {
                        for (; i <= normDim - vecSize; i += vecSize)
                        {
                            var v = ReadVector(ref pSrc, i);
                            WriteVector(ref pDst, i, v * rmsVec);
                        }
                    }

                    for (; i < normDim; i++)
                    {
                        Unsafe.Add(ref pDst, i) = Unsafe.Add(ref pSrc, i) * rms;
                    }
                }
            }

            if (batchRows > 1)
            {
                Parallel.For(0, batchRows, ProcessRow);
            }
            else
            {
                ProcessRow(0);
            }

            return result;
        }

        /// <summary>
        /// Layer Normalization (LayerNorm) widely used in Vision Transformers (ViT), CLIP, Stable Diffusion, and GPT.
        /// y = ((x - mean) / sqrt(var + eps)) * weight + bias
        /// </summary>
        public static Tensor<float> LayerNorm(Tensor<float> x, Tensor<float>? weight = null, Tensor<float>? bias = null, float eps = 1e-5f)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            if (x.Rank < 1) throw new ArgumentException("LayerNorm requires rank >= 1.", nameof(x));

            int normDim = x.Shape[x.Rank - 1];
            if (weight != null && (weight.Length != normDim || weight.Rank != 1))
            {
                throw new ArgumentException($"LayerNorm weight must be 1D with length {normDim}.", nameof(weight));
            }
            if (bias != null && (bias.Length != normDim || bias.Rank != 1))
            {
                throw new ArgumentException($"LayerNorm bias must be 1D with length {normDim}.", nameof(bias));
            }

            var contigX = x.IsContiguous ? x : x.ToContiguous();
            var contigW = weight != null ? (weight.IsContiguous ? weight : weight.ToContiguous()) : null;
            var contigB = bias != null ? (bias.IsContiguous ? bias : bias.ToContiguous()) : null;

            var result = new Tensor<float>(contigX.Shape);
            int batchRows = contigX.Length / normDim;

            float invNormDim = 1.0f / normDim;
            int vecSize = Vector<float>.Count;

            void ProcessRow(int r)
            {
                int offset = r * normDim;
                ref float pSrc = ref contigX.Storage.GetPinnableReference(contigX.Offset + offset);
                ref float pDst = ref result.Storage.GetPinnableReference(result.Offset + offset);

                // 1. Mean
                float sum = 0f;
                int i = 0;
                if (Vector.IsHardwareAccelerated && normDim >= vecSize)
                {
                    var sumVec = Vector<float>.Zero;
                    for (; i <= normDim - vecSize; i += vecSize)
                    {
                        sumVec += ReadVector(ref pSrc, i);
                    }
                    for (int k = 0; k < vecSize; k++) sum += sumVec[k];
                }
                for (; i < normDim; i++) sum += Unsafe.Add(ref pSrc, i);
                float mean = sum * invNormDim;

                // 2. Variance
                float sumVar = 0f;
                i = 0;
                var meanVec = new Vector<float>(mean);
                if (Vector.IsHardwareAccelerated && normDim >= vecSize)
                {
                    var varVec = Vector<float>.Zero;
                    for (; i <= normDim - vecSize; i += vecSize)
                    {
                        var diff = ReadVector(ref pSrc, i) - meanVec;
                        varVec += diff * diff;
                    }
                    for (int k = 0; k < vecSize; k++) sumVar += varVec[k];
                }
                for (; i < normDim; i++)
                {
                    float d = Unsafe.Add(ref pSrc, i) - mean;
                    sumVar += d * d;
                }

                float invStd = (float)(1.0 / Math.Sqrt(sumVar * invNormDim + eps));
                var invStdVec = new Vector<float>(invStd);

                // 3. Normalize, scale, bias
                i = 0;
                if (contigW != null && contigB != null)
                {
                    ref float pWeight = ref contigW.Storage.GetPinnableReference(contigW.Offset);
                    ref float pBias = ref contigB.Storage.GetPinnableReference(contigB.Offset);

                    if (Vector.IsHardwareAccelerated && normDim >= vecSize)
                    {
                        for (; i <= normDim - vecSize; i += vecSize)
                        {
                            var v = ReadVector(ref pSrc, i);
                            var w = ReadVector(ref pWeight, i);
                            var b = ReadVector(ref pBias, i);
                            WriteVector(ref pDst, i, ((v - meanVec) * invStdVec * w) + b);
                        }
                    }
                    for (; i < normDim; i++)
                    {
                        float val = Unsafe.Add(ref pSrc, i);
                        float w = Unsafe.Add(ref pWeight, i);
                        float b = Unsafe.Add(ref pBias, i);
                        Unsafe.Add(ref pDst, i) = ((val - mean) * invStd * w) + b;
                    }
                }
                else if (contigW != null)
                {
                    ref float pWeight = ref contigW.Storage.GetPinnableReference(contigW.Offset);
                    if (Vector.IsHardwareAccelerated && normDim >= vecSize)
                    {
                        for (; i <= normDim - vecSize; i += vecSize)
                        {
                            var v = ReadVector(ref pSrc, i);
                            var w = ReadVector(ref pWeight, i);
                            WriteVector(ref pDst, i, (v - meanVec) * invStdVec * w);
                        }
                    }
                    for (; i < normDim; i++)
                    {
                        float val = Unsafe.Add(ref pSrc, i);
                        float w = Unsafe.Add(ref pWeight, i);
                        Unsafe.Add(ref pDst, i) = (val - mean) * invStd * w;
                    }
                }
                else
                {
                    if (Vector.IsHardwareAccelerated && normDim >= vecSize)
                    {
                        for (; i <= normDim - vecSize; i += vecSize)
                        {
                            var v = ReadVector(ref pSrc, i);
                            WriteVector(ref pDst, i, (v - meanVec) * invStdVec);
                        }
                    }
                    for (; i < normDim; i++)
                    {
                        float val = Unsafe.Add(ref pSrc, i);
                        Unsafe.Add(ref pDst, i) = (val - mean) * invStd;
                    }
                }
            }

            if (batchRows > 1)
            {
                Parallel.For(0, batchRows, ProcessRow);
            }
            else
            {
                ProcessRow(0);
            }

            return result;
        }

        #endregion

        #region Activations: SiLU

        /// <summary>
        /// Sigmoid Linear Unit (SiLU / Swish): y = x * sigmoid(x) = x / (1 + exp(-x)).
        /// Essential for SwiGLU in modern LLMs and Diffusion models.
        /// </summary>
        public static Tensor<float> SiLU(Tensor<float> x)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            var contig = x.IsContiguous ? x : x.ToContiguous();
            var result = new Tensor<float>(contig.Shape);
            int len = contig.Length;

            Parallel.For(0, (len + 1023) / 1024, chunkIdx =>
            {
                int start = chunkIdx * 1024;
                int count = Math.Min(1024, len - start);
                ref float pSrc = ref contig.Storage.GetPinnableReference(contig.Offset + start);
                ref float pDst = ref result.Storage.GetPinnableReference(result.Offset + start);

                for (int i = 0; i < count; i++)
                {
                    float val = Unsafe.Add(ref pSrc, i);
                    Unsafe.Add(ref pDst, i) = val / (1.0f + (float)Math.Exp(-val));
                }
            });

            return result;
        }

        #endregion

        #region Attention: ScaledDotProductAttention (SDPA)

        /// <summary>
        /// Computes Scaled Dot-Product Attention: Softmax((Q @ K.T) * scale + mask) @ V.
        /// Supports multi-head batched tensors [B, H, S, D] or [B, S, D] or [S, D].
        /// </summary>
        /// <param name="q">Query tensor [..., S_q, D]</param>
        /// <param name="k">Key tensor [..., S_k, D]</param>
        /// <param name="v">Value tensor [..., S_k, D_v]</param>
        /// <param name="mask">Optional additive attention mask broadcastable to [..., S_q, S_k]</param>
        /// <param name="scale">Optional scaling factor (defaults to 1 / sqrt(D))</param>
        public static Tensor<float> ScaledDotProductAttention(
            Tensor<float> q,
            Tensor<float> k,
            Tensor<float> v,
            Tensor<float>? mask = null,
            float? scale = null)
        {
            if (q == null) throw new ArgumentNullException(nameof(q));
            if (k == null) throw new ArgumentNullException(nameof(k));
            if (v == null) throw new ArgumentNullException(nameof(v));

            if (q.Rank < 2 || k.Rank < 2 || v.Rank < 2)
            {
                throw new ArgumentException("Q, K, V tensors must have Rank >= 2.");
            }

            int d = q.Shape[q.Rank - 1];
            int kd = k.Shape[k.Rank - 1];
            if (d != kd)
            {
                throw new ArgumentException($"Query head dimension ({d}) must match Key head dimension ({kd}).");
            }

            float scaleFactor = scale ?? (1.0f / (float)Math.Sqrt(d));

            // 1. Transpose K along the last two dimensions: [..., D, S_k]
            var kT = k.Transpose(k.Rank - 2, k.Rank - 1);

            // 2. Q @ K.T -> [..., S_q, S_k]
            var scores = TensorBlas.MatMul(q, kT);

            // 3. Multiply by scale
            scores = scores * scaleFactor;

            // 4. Add mask if present
            if (mask != null)
            {
                scores = scores + mask;
            }

            // 5. Softmax along the last axis (-1)
            var attnWeights = SoftmaxFast(scores, axis: -1);

            // 6. Attn @ V -> [..., S_q, D_v]
            return TensorBlas.MatMul(attnWeights, v);
        }

        #endregion

        #region Computer Vision: Conv2D (im2col + GEMM Lowering)

        /// <summary>
        /// Performs 2D convolution: Input [N, C_in, H, W] * Weight [C_out, C_in, K_h, K_w] -> Output [N, C_out, H_out, W_out].
        /// Employs im2col lowering and SIMD GEMM for optimal execution without external native dependencies.
        /// </summary>
        public static Tensor<float> Conv2D(
            Tensor<float> input,
            Tensor<float> weight,
            Tensor<float>? bias = null,
            int stride = 1,
            int padding = 0,
            int dilation = 1)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (weight == null) throw new ArgumentNullException(nameof(weight));
            if (input.Rank != 4) throw new ArgumentException("Input tensor must have rank 4 [N, C_in, H, W].", nameof(input));
            if (weight.Rank != 4) throw new ArgumentException("Weight tensor must have rank 4 [C_out, C_in, K_h, K_w].", nameof(weight));

            int batchSize = input.Shape[0];
            int inChannels = input.Shape[1];
            int inH = input.Shape[2];
            int inW = input.Shape[3];

            int outChannels = weight.Shape[0];
            int weightInC = weight.Shape[1];
            int kH = weight.Shape[2];
            int kW = weight.Shape[3];

            if (inChannels != weightInC)
            {
                throw new ArgumentException($"Input channels ({inChannels}) does not match weight channels ({weightInC}).");
            }

            if (bias != null && (bias.Rank != 1 || bias.Length != outChannels))
            {
                throw new ArgumentException($"Bias must be 1D tensor with length {outChannels}.", nameof(bias));
            }

            int effKH = (kH - 1) * dilation + 1;
            int effKW = (kW - 1) * dilation + 1;

            int outH = (inH + 2 * padding - effKH) / stride + 1;
            int outW = (inW + 2 * padding - effKW) / stride + 1;

            if (outH <= 0 || outW <= 0)
            {
                throw new InvalidOperationException($"Calculated output spatial size [{outH}, {outW}] is invalid.");
            }

            var output = new Tensor<float>(batchSize, outChannels, outH, outW);

            // Reshape weights to 2D matrix: [outChannels, inChannels * kH * kW]
            int kernelSize = inChannels * kH * kW;
            int spatialOut = outH * outW;
            var weightMatrix = weight.Reshape(outChannels, kernelSize);

            // Process each batch image independently
            Parallel.For(0, batchSize, b =>
            {
                // 1. im2col: transform input image [C_in, inH, inW] into colMatrix [kernelSize, spatialOut]
                var colMatrix = new Tensor<float>(kernelSize, spatialOut);
                ref float pCol = ref colMatrix.Storage.GetPinnableReference(colMatrix.Offset);

                for (int c = 0; c < inChannels; c++)
                {
                    for (int kh = 0; kh < kH; kh++)
                    {
                        for (int kw = 0; kw < kW; kw++)
                        {
                            int colRow = (c * kH + kh) * kW + kw;
                            int colRowOffset = colRow * spatialOut;

                            for (int oh = 0; oh < outH; oh++)
                            {
                                int ih = oh * stride - padding + kh * dilation;
                                int outRowOffset = colRowOffset + oh * outW;

                                for (int ow = 0; ow < outW; ow++)
                                {
                                    int iw = ow * stride - padding + kw * dilation;
                                    if ((uint)ih < (uint)inH && (uint)iw < (uint)inW)
                                    {
                                        Unsafe.Add(ref pCol, outRowOffset + ow) = input[b, c, ih, iw];
                                    }
                                    else
                                    {
                                        Unsafe.Add(ref pCol, outRowOffset + ow) = 0f;
                                    }
                                }
                            }
                        }
                    }
                }

                // 2. GEMM: [outChannels, kernelSize] @ [kernelSize, spatialOut] -> [outChannels, spatialOut]
                var out2D = TensorBlas.MatMul2D(weightMatrix, colMatrix);
                ref float pOut2D = ref out2D.Storage.GetPinnableReference(out2D.Offset);

                // 3. Copy into output tensor and add bias if present
                for (int oc = 0; oc < outChannels; oc++)
                {
                    float bVal = bias != null ? bias[oc] : 0f;
                    int srcOffset = oc * spatialOut;

                    for (int oh = 0; oh < outH; oh++)
                    {
                        for (int ow = 0; ow < outW; ow++)
                        {
                            output[b, oc, oh, ow] = Unsafe.Add(ref pOut2D, srcOffset + oh * outW + ow) + bVal;
                        }
                    }
                }
            });

            return output;
        }

        #endregion
    }

    public static partial class Tensor
    {
        /// <summary>
        /// Computes numerically stable Softmax along the specified axis.
        /// </summary>
        public static Tensor<float> Softmax(Tensor<float> t, int axis = -1) => TensorOps.SoftmaxFast(t, axis);

        /// <summary>
        /// Computes Root Mean Square Normalization (RMSNorm) along the last axis.
        /// </summary>
        public static Tensor<float> RMSNorm(Tensor<float> x, Tensor<float>? weight = null, float eps = 1e-5f) =>
            TensorOps.RMSNorm(x, weight, eps);

        /// <summary>
        /// Computes Layer Normalization (LayerNorm) along the last axis.
        /// </summary>
        public static Tensor<float> LayerNorm(Tensor<float> x, Tensor<float>? weight = null, Tensor<float>? bias = null, float eps = 1e-5f) =>
            TensorOps.LayerNorm(x, weight, bias, eps);

        /// <summary>
        /// Computes Sigmoid Linear Unit (SiLU / Swish).
        /// </summary>
        public static Tensor<float> SiLU(Tensor<float> x) => TensorOps.SiLU(x);

        /// <summary>
        /// Computes Gaussian Error Linear Unit (GELU).
        /// </summary>
        public static Tensor<float> GELU(Tensor<float> x) => TensorOps.GELU(x);

        /// <summary>
        /// Computes Scaled Dot-Product Attention (SDPA).
        /// </summary>
        public static Tensor<float> ScaledDotProductAttention(
            Tensor<float> q,
            Tensor<float> k,
            Tensor<float> v,
            Tensor<float>? mask = null,
            float? scale = null) =>
            TensorOps.ScaledDotProductAttention(q, k, v, mask, scale);

        /// <summary>
        /// Computes 2D Convolution using im2col + GEMM lowering.
        /// </summary>
        public static Tensor<float> Conv2D(
            Tensor<float> input,
            Tensor<float> weight,
            Tensor<float>? bias = null,
            int stride = 1,
            int padding = 0,
            int dilation = 1) =>
            TensorOps.Conv2D(input, weight, bias, stride, padding, dilation);
    }
}
