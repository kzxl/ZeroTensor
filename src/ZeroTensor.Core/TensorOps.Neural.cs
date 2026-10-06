using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace ZeroTensor.Core
{
    /// <summary>
    /// Specifies the interpolation algorithm for spatial tensor resizing.
    /// </summary>
    public enum InterpolationMode
    {
        Nearest = 0,
        Bilinear = 1
    }

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

        #region LLM Primitives: EmbeddingLookup & RoPE

        /// <summary>
        /// Retrieves embedding vectors from a weight matrix: Output[..., D] = Weight[Indices[...], D].
        /// Supports multi-dimensional token batches [B, S] -> [B, S, D] or [S] -> [S, D].
        /// </summary>
        public static Tensor<float> EmbeddingLookup(Tensor<int> indices, Tensor<float> weight)
        {
            if (indices == null) throw new ArgumentNullException(nameof(indices));
            if (weight == null) throw new ArgumentNullException(nameof(weight));
            if (weight.Rank != 2) throw new ArgumentException("Embedding weight tensor must have rank 2 [VocabSize, EmbeddingDim].", nameof(weight));

            int vocabSize = weight.Shape[0];
            int embeddingDim = weight.Shape[1];

            // Build output shape: concat indices.Shape with [embeddingDim]
            var outDims = new int[indices.Rank + 1];
            for (int i = 0; i < indices.Rank; i++) outDims[i] = indices.Shape[i];
            outDims[outDims.Length - 1] = embeddingDim;

            var result = new Tensor<float>(new TensorShape(outDims));
            int numTokens = indices.Length;

            var contigIndices = indices.IsContiguous ? indices : indices.ToContiguous();
            var contigWeight = weight.IsContiguous ? weight : weight.ToContiguous();

            Parallel.For(0, numTokens, t =>
            {
                int tokenIdx = contigIndices.Storage.GetPinnableReference(contigIndices.Offset + t);
                if ((uint)tokenIdx >= (uint)vocabSize)
                {
                    throw new IndexOutOfRangeException($"Token index {tokenIdx} is out of vocabulary range [0, {vocabSize}).");
                }

                int srcOffset = contigWeight.Offset + tokenIdx * embeddingDim;
                int dstOffset = result.Offset + t * embeddingDim;

                ref float pSrc = ref contigWeight.Storage.GetPinnableReference(srcOffset);
                ref float pDst = ref result.Storage.GetPinnableReference(dstOffset);

                for (int d = 0; d < embeddingDim; d++)
                {
                    Unsafe.Add(ref pDst, d) = Unsafe.Add(ref pSrc, d);
                }
            });

            return result;
        }

        /// <summary>
        /// Applies Rotary Positional Embedding (RoPE) to Query or Key tensors.
        /// Supports standard LLaMA/Mistral/Qwen/DiT split-half layout or interleaved layout.
        /// x shape: [..., SeqLen, HeadDim].
        /// </summary>
        public static Tensor<float> ApplyRoPE(
            Tensor<float> x,
            int startPos = 0,
            float thetaBase = 10000.0f,
            bool interleaved = false)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            if (x.Rank < 2) throw new ArgumentException("Tensor x must have rank >= 2 [..., SeqLen, HeadDim].", nameof(x));

            int headDim = x.Shape[x.Rank - 1];
            int seqLen = x.Shape[x.Rank - 2];

            if ((headDim & 1) != 0)
            {
                throw new ArgumentException($"Head dimension ({headDim}) must be an even number for RoPE.", nameof(x));
            }

            var contigX = x.IsContiguous ? x : x.ToContiguous();
            var result = new Tensor<float>(contigX.Shape);

            int totalHeads = contigX.Length / (seqLen * headDim);
            int halfDim = headDim / 2;

            // Precompute inverse frequencies for halfDim
            float[] invFreq = new float[halfDim];
            for (int i = 0; i < halfDim; i++)
            {
                invFreq[i] = 1.0f / (float)Math.Pow(thetaBase, (2.0 * i) / headDim);
            }

            Parallel.For(0, totalHeads, h =>
            {
                int headBaseOffset = contigX.Offset + h * seqLen * headDim;
                int headDstBaseOffset = result.Offset + h * seqLen * headDim;

                for (int s = 0; s < seqLen; s++)
                {
                    int pos = startPos + s;
                    int tokenSrcOffset = headBaseOffset + s * headDim;
                    int tokenDstOffset = headDstBaseOffset + s * headDim;

                    ref float pSrc = ref contigX.Storage.GetPinnableReference(tokenSrcOffset);
                    ref float pDst = ref result.Storage.GetPinnableReference(tokenDstOffset);

                    if (!interleaved)
                    {
                        // Standard LLaMA / HuggingFace: split halves
                        for (int i = 0; i < halfDim; i++)
                        {
                            float angle = pos * invFreq[i];
                            float cos = (float)Math.Cos(angle);
                            float sin = (float)Math.Sin(angle);

                            float x1 = Unsafe.Add(ref pSrc, i);
                            float x2 = Unsafe.Add(ref pSrc, i + halfDim);

                            Unsafe.Add(ref pDst, i) = x1 * cos - x2 * sin;
                            Unsafe.Add(ref pDst, i + halfDim) = x2 * cos + x1 * sin;
                        }
                    }
                    else
                    {
                        // Interleaved GPT-NeoX layout
                        for (int i = 0; i < halfDim; i++)
                        {
                            float angle = pos * invFreq[i];
                            float cos = (float)Math.Cos(angle);
                            float sin = (float)Math.Sin(angle);

                            int idx1 = 2 * i;
                            int idx2 = 2 * i + 1;

                            float x1 = Unsafe.Add(ref pSrc, idx1);
                            float x2 = Unsafe.Add(ref pSrc, idx2);

                            Unsafe.Add(ref pDst, idx1) = x1 * cos - x2 * sin;
                            Unsafe.Add(ref pDst, idx2) = x2 * cos + x1 * sin;
                        }
                    }
                }
            });

            return result;
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

        #region Computer Vision: Pooling & Interpolation

        /// <summary>
        /// Max Pooling 2D: Downsamples spatial dimensions taking the maximum value in each sliding window.
        /// Input [N, C, H, W] -> Output [N, C, H_out, W_out].
        /// </summary>
        public static Tensor<float> MaxPool2D(
            Tensor<float> input,
            int kernelSize,
            int stride = -1,
            int padding = 0)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (input.Rank != 4) throw new ArgumentException("Input tensor must have rank 4 [N, C, H, W].", nameof(input));
            if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));

            int actualStride = stride <= 0 ? kernelSize : stride;
            int batchSize = input.Shape[0];
            int channels = input.Shape[1];
            int inH = input.Shape[2];
            int inW = input.Shape[3];

            int outH = (inH + 2 * padding - kernelSize) / actualStride + 1;
            int outW = (inW + 2 * padding - kernelSize) / actualStride + 1;

            if (outH <= 0 || outW <= 0)
            {
                throw new InvalidOperationException($"Calculated output spatial size [{outH}, {outW}] is invalid.");
            }

            var output = new Tensor<float>(batchSize, channels, outH, outW);

            Parallel.For(0, batchSize * channels, bc =>
            {
                int b = bc / channels;
                int c = bc % channels;

                for (int oh = 0; oh < outH; oh++)
                {
                    int startH = oh * actualStride - padding;
                    int endH = Math.Min(startH + kernelSize, inH);
                    int validStartH = Math.Max(startH, 0);

                    for (int ow = 0; ow < outW; ow++)
                    {
                        int startW = ow * actualStride - padding;
                        int endW = Math.Min(startW + kernelSize, inW);
                        int validStartW = Math.Max(startW, 0);

                        float maxVal = float.NegativeInfinity;

                        for (int ih = validStartH; ih < endH; ih++)
                        {
                            for (int iw = validStartW; iw < endW; iw++)
                            {
                                float val = input[b, c, ih, iw];
                                if (val > maxVal) maxVal = val;
                            }
                        }

                        output[b, c, oh, ow] = float.IsNegativeInfinity(maxVal) ? 0f : maxVal;
                    }
                }
            });

            return output;
        }

        /// <summary>
        /// Average Pooling 2D: Downsamples spatial dimensions taking the average value in each sliding window.
        /// Input [N, C, H, W] -> Output [N, C, H_out, W_out].
        /// </summary>
        public static Tensor<float> AvgPool2D(
            Tensor<float> input,
            int kernelSize,
            int stride = -1,
            int padding = 0,
            bool countIncludePad = true)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (input.Rank != 4) throw new ArgumentException("Input tensor must have rank 4 [N, C, H, W].", nameof(input));
            if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));

            int actualStride = stride <= 0 ? kernelSize : stride;
            int batchSize = input.Shape[0];
            int channels = input.Shape[1];
            int inH = input.Shape[2];
            int inW = input.Shape[3];

            int outH = (inH + 2 * padding - kernelSize) / actualStride + 1;
            int outW = (inW + 2 * padding - kernelSize) / actualStride + 1;

            if (outH <= 0 || outW <= 0)
            {
                throw new InvalidOperationException($"Calculated output spatial size [{outH}, {outW}] is invalid.");
            }

            var output = new Tensor<float>(batchSize, channels, outH, outW);
            float poolArea = kernelSize * kernelSize;

            Parallel.For(0, batchSize * channels, bc =>
            {
                int b = bc / channels;
                int c = bc % channels;

                for (int oh = 0; oh < outH; oh++)
                {
                    int startH = oh * actualStride - padding;
                    int endH = Math.Min(startH + kernelSize, inH);
                    int validStartH = Math.Max(startH, 0);

                    for (int ow = 0; ow < outW; ow++)
                    {
                        int startW = ow * actualStride - padding;
                        int endW = Math.Min(startW + kernelSize, inW);
                        int validStartW = Math.Max(startW, 0);

                        float sum = 0f;
                        int count = 0;

                        for (int ih = validStartH; ih < endH; ih++)
                        {
                            for (int iw = validStartW; iw < endW; iw++)
                            {
                                sum += input[b, c, ih, iw];
                                count++;
                            }
                        }

                        float divisor = countIncludePad ? poolArea : Math.Max(count, 1);
                        output[b, c, oh, ow] = sum / divisor;
                    }
                }
            });

            return output;
        }

        /// <summary>
        /// Interpolates / resizes a 4D spatial tensor [N, C, H, W] to target [N, C, outH, outW].
        /// Supports Bilinear and Nearest neighbor interpolation with corner alignment.
        /// </summary>
        public static Tensor<float> Interpolate2D(
            Tensor<float> input,
            int outH,
            int outW,
            InterpolationMode mode = InterpolationMode.Bilinear,
            bool alignCorners = false)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (input.Rank != 4) throw new ArgumentException("Input tensor must have rank 4 [N, C, H, W].", nameof(input));
            if (outH <= 0) throw new ArgumentOutOfRangeException(nameof(outH));
            if (outW <= 0) throw new ArgumentOutOfRangeException(nameof(outW));

            int batchSize = input.Shape[0];
            int channels = input.Shape[1];
            int inH = input.Shape[2];
            int inW = input.Shape[3];

            if (inH == outH && inW == outW)
            {
                return input.Clone();
            }

            var output = new Tensor<float>(batchSize, channels, outH, outW);

            if (mode == InterpolationMode.Nearest)
            {
                float scaleH = (float)inH / outH;
                float scaleW = (float)inW / outW;

                Parallel.For(0, batchSize * channels, bc =>
                {
                    int b = bc / channels;
                    int c = bc % channels;

                    for (int oh = 0; oh < outH; oh++)
                    {
                        int ih = Math.Min((int)(oh * scaleH), inH - 1);
                        for (int ow = 0; ow < outW; ow++)
                        {
                            int iw = Math.Min((int)(ow * scaleW), inW - 1);
                            output[b, c, oh, ow] = input[b, c, ih, iw];
                        }
                    }
                });

                return output;
            }

            // Bilinear mode
            float rH = alignCorners && outH > 1 ? (float)(inH - 1) / (outH - 1) : (float)inH / outH;
            float rW = alignCorners && outW > 1 ? (float)(inW - 1) / (outW - 1) : (float)inW / outW;

            Parallel.For(0, batchSize * channels, bc =>
            {
                int b = bc / channels;
                int c = bc % channels;

                for (int oh = 0; oh < outH; oh++)
                {
                    float srcH = alignCorners ? oh * rH : (oh + 0.5f) * rH - 0.5f;
                    int h0 = (int)Math.Floor(srcH);
                    int h1 = Math.Min(h0 + 1, inH - 1);
                    h0 = Math.Max(h0, 0);
                    float dh = srcH - h0;
                    if (srcH < 0f) dh = 0f;

                    for (int ow = 0; ow < outW; ow++)
                    {
                        float srcW = alignCorners ? ow * rW : (ow + 0.5f) * rW - 0.5f;
                        int w0 = (int)Math.Floor(srcW);
                        int w1 = Math.Min(w0 + 1, inW - 1);
                        w0 = Math.Max(w0, 0);
                        float dw = srcW - w0;
                        if (srcW < 0f) dw = 0f;

                        float p00 = input[b, c, h0, w0];
                        float p01 = input[b, c, h0, w1];
                        float p10 = input[b, c, h1, w0];
                        float p11 = input[b, c, h1, w1];

                        float val = (1f - dh) * (1f - dw) * p00 +
                                    (1f - dh) * dw * p01 +
                                    dh * (1f - dw) * p10 +
                                    dh * dw * p11;

                        output[b, c, oh, ow] = val;
                    }
                }
            });

            return output;
        }

        #endregion

        #region Normalization: GroupNorm

        /// <summary>
        /// Group Normalization: divides channels into G groups and normalizes spatial features independently.
        /// Essential for Diffusion models (Stable Diffusion, SDXL, SANA, DiT), VAE decoders, and ConvNets.
        /// y = ((x - mean) / sqrt(var + eps)) * weight + bias
        /// </summary>
        public static Tensor<float> GroupNorm(
            Tensor<float> x,
            int numGroups,
            Tensor<float>? weight = null,
            Tensor<float>? bias = null,
            float eps = 1e-5f)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            if (x.Rank != 4) throw new ArgumentException("GroupNorm requires rank 4 [N, C, H, W].", nameof(x));
            if (numGroups <= 0) throw new ArgumentOutOfRangeException(nameof(numGroups));

            int batchSize = x.Shape[0];
            int channels = x.Shape[1];
            int h = x.Shape[2];
            int w = x.Shape[3];

            if (channels % numGroups != 0)
            {
                throw new ArgumentException($"Channel count ({channels}) must be divisible by numGroups ({numGroups}).");
            }

            if (weight != null && (weight.Length != channels || weight.Rank != 1))
            {
                throw new ArgumentException($"Weight must be 1D tensor with length {channels}.", nameof(weight));
            }

            if (bias != null && (bias.Length != channels || bias.Rank != 1))
            {
                throw new ArgumentException($"Bias must be 1D tensor with length {channels}.", nameof(bias));
            }

            int channelsPerGroup = channels / numGroups;
            int groupSpatialSize = channelsPerGroup * h * w;
            float invGroupSize = 1.0f / groupSpatialSize;

            var contigX = x.IsContiguous ? x : x.ToContiguous();
            var result = new Tensor<float>(contigX.Shape);

            Parallel.For(0, batchSize * numGroups, bg =>
            {
                int b = bg / numGroups;
                int g = bg % numGroups;
                int startChannel = g * channelsPerGroup;
                int endChannel = startChannel + channelsPerGroup;

                // 1. Calculate Mean
                float sum = 0f;
                for (int c = startChannel; c < endChannel; c++)
                {
                    for (int ih = 0; ih < h; ih++)
                    {
                        for (int iw = 0; iw < w; iw++)
                        {
                            sum += contigX[b, c, ih, iw];
                        }
                    }
                }
                float mean = sum * invGroupSize;

                // 2. Calculate Variance
                float sumVar = 0f;
                for (int c = startChannel; c < endChannel; c++)
                {
                    for (int ih = 0; ih < h; ih++)
                    {
                        for (int iw = 0; iw < w; iw++)
                        {
                            float diff = contigX[b, c, ih, iw] - mean;
                            sumVar += diff * diff;
                        }
                    }
                }
                float invStd = (float)(1.0 / Math.Sqrt(sumVar * invGroupSize + eps));

                // 3. Normalize, scale, bias
                for (int c = startChannel; c < endChannel; c++)
                {
                    float wVal = weight != null ? weight[c] : 1f;
                    float bVal = bias != null ? bias[c] : 0f;

                    for (int ih = 0; ih < h; ih++)
                    {
                        for (int iw = 0; iw < w; iw++)
                        {
                            float normVal = (contigX[b, c, ih, iw] - mean) * invStd;
                            result[b, c, ih, iw] = normVal * wVal + bVal;
                        }
                    }
                }
            });

            return result;
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

        /// <summary>
        /// Retrieves embedding vectors from a weight matrix: Output[..., D] = Weight[Indices[...], D].
        /// </summary>
        public static Tensor<float> EmbeddingLookup(Tensor<int> indices, Tensor<float> weight) =>
            TensorOps.EmbeddingLookup(indices, weight);

        /// <summary>
        /// Applies Rotary Positional Embedding (RoPE) to Query or Key tensors.
        /// </summary>
        public static Tensor<float> ApplyRoPE(
            Tensor<float> x,
            int startPos = 0,
            float thetaBase = 10000.0f,
            bool interleaved = false) =>
            TensorOps.ApplyRoPE(x, startPos, thetaBase, interleaved);

        /// <summary>
        /// Max Pooling 2D downsampling.
        /// </summary>
        public static Tensor<float> MaxPool2D(
            Tensor<float> input,
            int kernelSize,
            int stride = -1,
            int padding = 0) =>
            TensorOps.MaxPool2D(input, kernelSize, stride, padding);

        /// <summary>
        /// Average Pooling 2D downsampling.
        /// </summary>
        public static Tensor<float> AvgPool2D(
            Tensor<float> input,
            int kernelSize,
            int stride = -1,
            int padding = 0,
            bool countIncludePad = true) =>
            TensorOps.AvgPool2D(input, kernelSize, stride, padding, countIncludePad);

        /// <summary>
        /// Spatial 2D interpolation / resizing (Bilinear or Nearest).
        /// </summary>
        public static Tensor<float> Interpolate2D(
            Tensor<float> input,
            int outH,
            int outW,
            InterpolationMode mode = InterpolationMode.Bilinear,
            bool alignCorners = false) =>
            TensorOps.Interpolate2D(input, outH, outW, mode, alignCorners);

        /// <summary>
        /// Group Normalization across G channel groups.
        /// </summary>
        public static Tensor<float> GroupNorm(
            Tensor<float> x,
            int numGroups,
            Tensor<float>? weight = null,
            Tensor<float>? bias = null,
            float eps = 1e-5f) =>
            TensorOps.GroupNorm(x, numGroups, weight, bias, eps);
    }
}
