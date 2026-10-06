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

        #region Activations: SwiGLU & GeGLU (Fused Streaming Kernels)

        /// <summary>
        /// Fused SwiGLU (Swish Gated Linear Unit): y = SiLU(gate) * up = (gate / (1 + exp(-gate))) * up.
        /// Evaluated in a single-pass streaming kernel with zero intermediate buffer allocations.
        /// Essential for high-throughput LLaMA 2/3, Mistral, Qwen, DeepSeek, and Mixtral feed-forward networks.
        /// </summary>
        public static Tensor<float> SwiGLU(Tensor<float> gate, Tensor<float> up)
        {
            if (gate == null) throw new ArgumentNullException(nameof(gate));
            if (up == null) throw new ArgumentNullException(nameof(up));
            if (!gate.Shape.Equals(up.Shape))
                throw new ArgumentException($"Gate shape ({gate.Shape}) and Up shape ({up.Shape}) must match exactly for SwiGLU.");

            var contigGate = gate.IsContiguous ? gate : gate.ToContiguous();
            var contigUp = up.IsContiguous ? up : up.ToContiguous();

            var result = new Tensor<float>(contigGate.Shape);
            int len = contigGate.Length;

            unsafe
            {
                fixed (float* pG = &contigGate.Storage.GetPinnableReference(contigGate.Offset))
                fixed (float* pU = &contigUp.Storage.GetPinnableReference(contigUp.Offset))
                fixed (float* pDst = &result.Storage.GetPinnableReference(result.Offset))
                {
                    IntPtr ptrG = (IntPtr)pG;
                    IntPtr ptrU = (IntPtr)pU;
                    IntPtr ptrDst = (IntPtr)pDst;

                    int chunkSize = 2048;
                    int numChunks = (len + chunkSize - 1) / chunkSize;

                    Parallel.For(0, numChunks, chunkIdx =>
                    {
                        float* localG = (float*)ptrG;
                        float* localU = (float*)ptrU;
                        float* localDst = (float*)ptrDst;

                        int start = chunkIdx * chunkSize;
                        int count = Math.Min(chunkSize, len - start);

                        for (int i = 0; i < count; i++)
                        {
                            int idx = start + i;
                            float g = localG[idx];
                            float u = localU[idx];
                            float siluG = g / (1.0f + (float)Math.Exp(-g));
                            localDst[idx] = siluG * u;
                        }
                    });
                }
            }

            return result;
        }

        /// <summary>
        /// Single-tensor Fused SwiGLU: splits tensor along <paramref name="dim"/> into [gate, up] and computes SwiGLU(gate, up).
        /// </summary>
        public static Tensor<float> SwiGLU(Tensor<float> x, int dim = -1)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            int ax = dim < 0 ? dim + x.Rank : dim;
            if (ax < 0 || ax >= x.Rank) throw new ArgumentOutOfRangeException(nameof(dim));

            int totalDim = x.Shape[ax];
            if (totalDim % 2 != 0)
                throw new ArgumentException($"Target dimension {ax} size ({totalDim}) must be even to split into gate and up projections.");

            int half = totalDim / 2;
            var gate = x.Slice(ax, 0, half);
            var up = x.Slice(ax, half, half);
            return SwiGLU(gate, up);
        }

        /// <summary>
        /// Fused GeGLU (Gaussian Error Gated Linear Unit): y = GELU(gate) * up.
        /// Uses fast tanh approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
        /// Evaluated in a single-pass streaming kernel with zero intermediate buffer allocations.
        /// Essential for Gemma, PaLM, and Diffusion Transformer (DiT / SANA) MLP blocks.
        /// </summary>
        public static Tensor<float> GeGLU(Tensor<float> gate, Tensor<float> up)
        {
            if (gate == null) throw new ArgumentNullException(nameof(gate));
            if (up == null) throw new ArgumentNullException(nameof(up));
            if (!gate.Shape.Equals(up.Shape))
                throw new ArgumentException($"Gate shape ({gate.Shape}) and Up shape ({up.Shape}) must match exactly for GeGLU.");

            var contigGate = gate.IsContiguous ? gate : gate.ToContiguous();
            var contigUp = up.IsContiguous ? up : up.ToContiguous();

            var result = new Tensor<float>(contigGate.Shape);
            int len = contigGate.Length;

            const float sqrt2OverPi = 0.7978845608028654f;
            const float geluCoeff = 0.044715f;

            unsafe
            {
                fixed (float* pG = &contigGate.Storage.GetPinnableReference(contigGate.Offset))
                fixed (float* pU = &contigUp.Storage.GetPinnableReference(contigUp.Offset))
                fixed (float* pDst = &result.Storage.GetPinnableReference(result.Offset))
                {
                    IntPtr ptrG = (IntPtr)pG;
                    IntPtr ptrU = (IntPtr)pU;
                    IntPtr ptrDst = (IntPtr)pDst;

                    int chunkSize = 2048;
                    int numChunks = (len + chunkSize - 1) / chunkSize;

                    Parallel.For(0, numChunks, chunkIdx =>
                    {
                        float* localG = (float*)ptrG;
                        float* localU = (float*)ptrU;
                        float* localDst = (float*)ptrDst;

                        int start = chunkIdx * chunkSize;
                        int count = Math.Min(chunkSize, len - start);

                        for (int i = 0; i < count; i++)
                        {
                            int idx = start + i;
                            float g = localG[idx];
                            float u = localU[idx];
                            float inner = sqrt2OverPi * (g + geluCoeff * g * g * g);
                            float geluG = 0.5f * g * (1.0f + (float)Math.Tanh(inner));
                            localDst[idx] = geluG * u;
                        }
                    });
                }
            }

            return result;
        }

        /// <summary>
        /// Single-tensor Fused GeGLU: splits tensor along <paramref name="dim"/> into [gate, up] and computes GeGLU(gate, up).
        /// </summary>
        public static Tensor<float> GeGLU(Tensor<float> x, int dim = -1)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            int ax = dim < 0 ? dim + x.Rank : dim;
            if (ax < 0 || ax >= x.Rank) throw new ArgumentOutOfRangeException(nameof(dim));

            int totalDim = x.Shape[ax];
            if (totalDim % 2 != 0)
                throw new ArgumentException($"Target dimension {ax} size ({totalDim}) must be even to split into gate and up projections.");

            int half = totalDim / 2;
            var gate = x.Slice(ax, 0, half);
            var up = x.Slice(ax, half, half);
            return GeGLU(gate, up);
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

        #region Attention: FlashAttentionCpu (Tiled Online Softmax with SIMD & GQA)

        /// <summary>
        /// Tiled Online Softmax Attention (FlashAttention for CPU).
        /// Computes Softmax(Q @ K.T * scale + mask) @ V without materializing the O(N^2) attention score matrix.
        /// Dramatically cuts memory footprint from O(N^2) to O(1) and maximizes L1/L2 CPU cache residency for long-context LLMs.
        /// Fully accelerated with SIMD Vector<float> and natively supports Grouped Query Attention (GQA / MQA).
        /// </summary>
        public static Tensor<float> FlashAttentionCpu(
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
                throw new ArgumentException("Q, K, V tensors must have Rank >= 2.");

            int qRank = q.Rank;
            int sQ = q.Shape[qRank - 2];
            int d = q.Shape[qRank - 1];
            int sK = k.Shape[k.Rank - 2];
            int dK = k.Shape[k.Rank - 1];
            int dV = v.Shape[v.Rank - 1];

            if (d != dK)
                throw new ArgumentException($"Query head dim ({d}) must match Key head dim ({dK}).");

            float scaleFactor = scale ?? (1.0f / (float)Math.Sqrt(d));

            // Support Grouped Query Attention (GQA / MQA) for 4D tensors [Batch, Heads, SeqLen, Dim]
            bool isGqa = false;
            int gqaGroup = 1;
            int hQ = 1;
            int hKV = 1;

            if (qRank == 4 && k.Rank == 4)
            {
                hQ = q.Shape[1];
                hKV = k.Shape[1];
                if (hQ % hKV != 0)
                    throw new ArgumentException($"Query heads ({hQ}) must be divisible by Key/Value heads ({hKV}) for GQA.");
                gqaGroup = hQ / hKV;
                isGqa = gqaGroup > 1;
            }

            // Compute total batch count (heads * batch elements)
            int batchCount = 1;
            for (int i = 0; i < qRank - 2; i++) batchCount *= q.Shape[i];

            // Output shape is same as Q except last dimension is dV
            var outDims = new int[qRank];
            for (int i = 0; i < qRank - 1; i++) outDims[i] = q.Shape[i];
            outDims[qRank - 1] = dV;
            var result = new Tensor<float>(outDims);

            var qContig = q.ToContiguous();
            var kContig = k.ToContiguous();
            var vContig = v.ToContiguous();
            var maskContig = mask?.ToContiguous();

            int qBatchStride = sQ * d;
            int kBatchStride = sK * d;
            int vBatchStride = sK * dV;
            int outBatchStride = sQ * dV;
            int maskBatchStride = sQ * sK;

            int vecSize = Vector<float>.Count;

            unsafe
            {
                fixed (float* pQ = &qContig.Storage.GetPinnableReference(qContig.Offset))
                fixed (float* pK = &kContig.Storage.GetPinnableReference(kContig.Offset))
                fixed (float* pV = &vContig.Storage.GetPinnableReference(vContig.Offset))
                fixed (float* pOut = &result.Storage.GetPinnableReference(result.Offset))
                {
                    IntPtr ptrQ = (IntPtr)pQ;
                    IntPtr ptrK = (IntPtr)pK;
                    IntPtr ptrV = (IntPtr)pV;
                    IntPtr ptrOut = (IntPtr)pOut;
                    IntPtr ptrMask = IntPtr.Zero;

                    Action runParallel = () =>
                    {
                        Parallel.For(0, batchCount, b =>
                        {
                            float* localQ = (float*)ptrQ;
                            float* localK = (float*)ptrK;
                            float* localV = (float*)ptrV;
                            float* localOut = (float*)ptrOut;
                            float* localMask = (float*)ptrMask;

                            int bKV = b;
                            if (isGqa)
                            {
                                int bSeq = b / hQ;
                                int curHQ = b % hQ;
                                int curHKV = curHQ / gqaGroup;
                                bKV = bSeq * hKV + curHKV;
                            }

                            int qBatchOff = b * qBatchStride;
                            int kBatchOff = bKV * kBatchStride;
                            int vBatchOff = bKV * vBatchStride;
                            int outBatchOff = b * outBatchStride;
                            int maskBatchOff = b * maskBatchStride;

                            Span<float> acc = stackalloc float[Math.Min(dV, 512)];
                            float[]? heapAcc = null;
                            if (dV > 512)
                            {
                                heapAcc = new float[dV];
                                acc = heapAcc.AsSpan();
                            }

                            fixed (float* pAcc = acc)
                            {
                                for (int i = 0; i < sQ; i++)
                                {
                                    int qOff = qBatchOff + i * d;
                                    int outOff = outBatchOff + i * dV;
                                    int maskRowOff = localMask != null ? (maskBatchStride > 0 ? (b % (maskContig!.Length / maskBatchStride)) * maskBatchStride + i * sK : 0) : 0;

                                    float maxScore = float.NegativeInfinity;
                                    float sumExp = 0f;
                                    acc.Clear();

                                    for (int j = 0; j < sK; j++)
                                    {
                                        int kOff = kBatchOff + j * d;
                                        int vOff = vBatchOff + j * dV;

                                        // 1. SIMD-accelerated Dot Product: Q[i] . K[j]
                                        float dot = 0f;
                                        int c = 0;
                                        if (Vector.IsHardwareAccelerated && d >= vecSize)
                                        {
                                            var vDot = Vector<float>.Zero;
                                            for (; c <= d - vecSize; c += vecSize)
                                            {
                                                var vQ = Unsafe.ReadUnaligned<Vector<float>>(localQ + qOff + c);
                                                var vK = Unsafe.ReadUnaligned<Vector<float>>(localK + kOff + c);
                                                vDot += vQ * vK;
                                            }
                                            for (int kIdx = 0; kIdx < vecSize; kIdx++) dot += vDot[kIdx];
                                        }
                                        for (; c < d; c++)
                                        {
                                            dot += localQ[qOff + c] * localK[kOff + c];
                                        }

                                        float score = dot * scaleFactor;
                                        if (localMask != null)
                                        {
                                            score += localMask[maskRowOff + j];
                                        }

                                        // 2. Online Softmax & Value accumulation with SIMD Vector update
                                        if (score > maxScore)
                                        {
                                            float alpha = (float)Math.Exp(maxScore - score);
                                            sumExp = sumExp * alpha + 1f;

                                            var vAlpha = new Vector<float>(alpha);
                                            int cv = 0;
                                            if (Vector.IsHardwareAccelerated && dV >= vecSize)
                                            {
                                                for (; cv <= dV - vecSize; cv += vecSize)
                                                {
                                                    var vA = Unsafe.ReadUnaligned<Vector<float>>(pAcc + cv);
                                                    var vVal = Unsafe.ReadUnaligned<Vector<float>>(localV + vOff + cv);
                                                    Unsafe.WriteUnaligned(pAcc + cv, vA * vAlpha + vVal);
                                                }
                                            }
                                            for (; cv < dV; cv++)
                                            {
                                                pAcc[cv] = pAcc[cv] * alpha + localV[vOff + cv];
                                            }

                                            maxScore = score;
                                        }
                                        else
                                        {
                                            float p = (float)Math.Exp(score - maxScore);
                                            sumExp += p;

                                            var vP = new Vector<float>(p);
                                            int cv = 0;
                                            if (Vector.IsHardwareAccelerated && dV >= vecSize)
                                            {
                                                for (; cv <= dV - vecSize; cv += vecSize)
                                                {
                                                    var vA = Unsafe.ReadUnaligned<Vector<float>>(pAcc + cv);
                                                    var vVal = Unsafe.ReadUnaligned<Vector<float>>(localV + vOff + cv);
                                                    Unsafe.WriteUnaligned(pAcc + cv, vA + vP * vVal);
                                                }
                                            }
                                            for (; cv < dV; cv++)
                                            {
                                                pAcc[cv] += p * localV[vOff + cv];
                                            }
                                        }
                                    }

                                    // 3. Normalize output with invSum via SIMD
                                    float invSum = sumExp > 0f ? 1.0f / sumExp : 0f;
                                    var vInv = new Vector<float>(invSum);
                                    int cOut = 0;
                                    if (Vector.IsHardwareAccelerated && dV >= vecSize)
                                    {
                                        for (; cOut <= dV - vecSize; cOut += vecSize)
                                        {
                                            var vA = Unsafe.ReadUnaligned<Vector<float>>(pAcc + cOut);
                                            Unsafe.WriteUnaligned(localOut + outOff + cOut, vA * vInv);
                                        }
                                    }
                                    for (; cOut < dV; cOut++)
                                    {
                                        localOut[outOff + cOut] = pAcc[cOut] * invSum;
                                    }
                                }
                            }
                        });
                    };

                    if (maskContig != null)
                    {
                        fixed (float* pMask = &maskContig.Storage.GetPinnableReference(maskContig.Offset))
                        {
                            ptrMask = (IntPtr)pMask;
                            runParallel();
                        }
                    }
                    else
                    {
                        runParallel();
                    }
                }
            }

            return result;
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
        /// Applies Rotary Positional Embedding (RoPE) in-place to Query or Key tensors directly without memory allocations.
        /// Precomputes sinusoidal rotary angles once across all heads for maximum decoding speed.
        /// </summary>
        public static void ApplyRoPEInPlace(
            this Tensor<float> x,
            int startPos = 0,
            float thetaBase = 10000.0f,
            bool interleaved = false)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            if (x.Rank < 2) throw new ArgumentException("Tensor x must have rank >= 2 [..., SeqLen, HeadDim].", nameof(x));

            int headDim = x.Shape[x.Rank - 1];
            int seqLen = x.Shape[x.Rank - 2];

            if ((headDim & 1) != 0)
                throw new ArgumentException($"Head dimension ({headDim}) must be an even number for RoPE.", nameof(x));

            if (!x.IsContiguous)
                throw new InvalidOperationException("In-place RoPE currently requires contiguous tensor memory.");

            int totalHeads = x.Length / (seqLen * headDim);
            int halfDim = headDim / 2;
            int totalTableElems = seqLen * halfDim;

            // Precompute cos & sin table across all sequence positions ONCE for all heads
            Span<float> cosTable = stackalloc float[Math.Min(totalTableElems, 2048)];
            Span<float> sinTable = stackalloc float[Math.Min(totalTableElems, 2048)];
            float[]? heapCos = null;
            float[]? heapSin = null;

            if (totalTableElems > 2048)
            {
                heapCos = System.Buffers.ArrayPool<float>.Shared.Rent(totalTableElems);
                heapSin = System.Buffers.ArrayPool<float>.Shared.Rent(totalTableElems);
                cosTable = heapCos.AsSpan(0, totalTableElems);
                sinTable = heapSin.AsSpan(0, totalTableElems);
            }

            try
            {
                // Precompute table
                for (int s = 0; s < seqLen; s++)
                {
                    int pos = startPos + s;
                    int sOffset = s * halfDim;
                    for (int i = 0; i < halfDim; i++)
                    {
                        float invFreq = 1.0f / (float)Math.Pow(thetaBase, (2.0 * i) / headDim);
                        float angle = pos * invFreq;
                        cosTable[sOffset + i] = (float)Math.Cos(angle);
                        sinTable[sOffset + i] = (float)Math.Sin(angle);
                    }
                }

                unsafe
                {
                    fixed (float* pX = &x.Storage.GetPinnableReference(x.Offset))
                    fixed (float* pCos = cosTable)
                    fixed (float* pSin = sinTable)
                    {
                        IntPtr ptrX = (IntPtr)pX;
                        IntPtr ptrCos = (IntPtr)pCos;
                        IntPtr ptrSin = (IntPtr)pSin;

                        Parallel.For(0, totalHeads, h =>
                        {
                            float* localX = (float*)ptrX;
                            float* localCos = (float*)ptrCos;
                            float* localSin = (float*)ptrSin;

                            int headBaseOffset = h * seqLen * headDim;

                            for (int s = 0; s < seqLen; s++)
                            {
                                int tokenOffset = headBaseOffset + s * headDim;
                                int sOffset = s * halfDim;

                                if (!interleaved)
                                {
                                    for (int i = 0; i < halfDim; i++)
                                    {
                                        float cos = localCos[sOffset + i];
                                        float sin = localSin[sOffset + i];

                                        float x1 = localX[tokenOffset + i];
                                        float x2 = localX[tokenOffset + i + halfDim];

                                        localX[tokenOffset + i] = x1 * cos - x2 * sin;
                                        localX[tokenOffset + i + halfDim] = x2 * cos + x1 * sin;
                                    }
                                }
                                else
                                {
                                    for (int i = 0; i < halfDim; i++)
                                    {
                                        float cos = localCos[sOffset + i];
                                        float sin = localSin[sOffset + i];

                                        int idx1 = 2 * i;
                                        int idx2 = 2 * i + 1;

                                        float x1 = localX[tokenOffset + idx1];
                                        float x2 = localX[tokenOffset + idx2];

                                        localX[tokenOffset + idx1] = x1 * cos - x2 * sin;
                                        localX[tokenOffset + idx2] = x2 * cos + x1 * sin;
                                    }
                                }
                            }
                        });
                    }
                }
            }
            finally
            {
                if (heapCos != null) System.Buffers.ArrayPool<float>.Shared.Return(heapCos);
                if (heapSin != null) System.Buffers.ArrayPool<float>.Shared.Return(heapSin);
            }
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
            var result = x.Clone();
            ApplyRoPEInPlace(result, startPos, thetaBase, interleaved);
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

            int batchElements = outChannels * spatialOut;
            int vecSize = Vector<float>.Count;

            // Process each batch image independently
            Parallel.For(0, batchSize, b =>
            {
                // 1. im2col: rent memory from TensorPool to avoid massive GC heap allocations
                using var rentedCol = TensorPool.Rent<float>(kernelSize, spatialOut);
                var colMatrix = rentedCol.Tensor;
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

                // 2. Direct Zero-Allocation GEMM into output buffer: [outChannels, kernelSize] @ [kernelSize, spatialOut] -> output[b, ...]
                int batchOffset = output.Offset + b * batchElements;
                var out2DSlice = new Tensor<float>(output.Storage, batchOffset, new TensorShape(outChannels, spatialOut), new[] { spatialOut, 1 });
                TensorBlas.Gemm(weightMatrix, colMatrix, out2DSlice, 1.0f, 0.0f);

                // 3. Vectorized Bias Addition if present
                if (bias != null)
                {
                    ref float pBias = ref bias.Storage.GetPinnableReference(bias.Offset);
                    ref float pOut = ref output.Storage.GetPinnableReference(batchOffset);

                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        float bVal = Unsafe.Add(ref pBias, oc);
                        if (bVal == 0f) continue;

                        int rowOff = oc * spatialOut;
                        var vBias = new Vector<float>(bVal);
                        int p = 0;

                        if (Vector.IsHardwareAccelerated && spatialOut >= vecSize)
                        {
                            for (; p <= spatialOut - vecSize; p += vecSize)
                            {
                                var v = ReadVector(ref pOut, rowOff + p);
                                WriteVector(ref pOut, rowOff + p, v + vBias);
                            }
                        }

                        for (; p < spatialOut; p++)
                        {
                            Unsafe.Add(ref pOut, rowOff + p) += bVal;
                        }
                    }
                }
            });

            return output;
        }

        /// <summary>
        /// Performs 1D Convolution: Input [N, C_in, L] * Weight [C_out, C_in, K] -> Output [N, C_out, L_out].
        /// Slices spatial 1D sequences via im2col lowering and SIMD GEMM. Essential for Speech/Audio (Whisper) and sequential signals.
        /// </summary>
        public static Tensor<float> Conv1D(
            Tensor<float> input,
            Tensor<float> weight,
            Tensor<float>? bias = null,
            int stride = 1,
            int padding = 0,
            int dilation = 1)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (weight == null) throw new ArgumentNullException(nameof(weight));
            if (input.Rank != 3) throw new ArgumentException("Input tensor must have rank 3 [N, C_in, L].", nameof(input));
            if (weight.Rank != 3) throw new ArgumentException("Weight tensor must have rank 3 [C_out, C_in, K].", nameof(weight));

            int batchSize = input.Shape[0];
            int inChannels = input.Shape[1];
            int inL = input.Shape[2];

            int outChannels = weight.Shape[0];
            int inChannelsW = weight.Shape[1];
            int kernelSize = weight.Shape[2];

            if (inChannels != inChannelsW)
                throw new InvalidOperationException($"Channel mismatch: Input C_in={inChannels}, Weight C_in={inChannelsW}.");

            int effectiveK = dilation * (kernelSize - 1) + 1;
            int outL = (inL + 2 * padding - effectiveK) / stride + 1;
            if (outL <= 0)
                throw new InvalidOperationException($"Computed output length {outL} is non-positive.");

            var output = new Tensor<float>(batchSize, outChannels, outL);
            var weightMatrix = weight.Reshape(outChannels, inChannels * kernelSize);

            var inContig = input.ToContiguous();

            unsafe
            {
                fixed (float* pIn = &inContig.Storage.GetPinnableReference(inContig.Offset))
                {
                    IntPtr ptrIn = (IntPtr)pIn;
                    Parallel.For(0, batchSize, b =>
                    {
                        float* localIn = (float*)ptrIn;
                        var colMatrix = new Tensor<float>(inChannels * kernelSize, outL);
                        ref float pCol = ref colMatrix.Storage.GetPinnableReference(0);

                        int batchOffset = b * inChannels * inL;

                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            int inChannelOffset = batchOffset + ic * inL;

                            for (int k = 0; k < kernelSize; k++)
                            {
                                int rowCol = ic * kernelSize + k;
                                int outRowOffset = rowCol * outL;

                                for (int ol = 0; ol < outL; ol++)
                                {
                                    int inIdx = ol * stride - padding + k * dilation;
                                    if (inIdx >= 0 && inIdx < inL)
                                    {
                                        Unsafe.Add(ref pCol, outRowOffset + ol) = localIn[inChannelOffset + inIdx];
                                    }
                                    else
                                    {
                                        Unsafe.Add(ref pCol, outRowOffset + ol) = 0f;
                                    }
                                }
                            }
                        }

                        var out2D = TensorBlas.MatMul2D(weightMatrix, colMatrix);
                        ref float pOut2D = ref out2D.Storage.GetPinnableReference(out2D.Offset);

                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            float bVal = bias != null ? bias[oc] : 0f;
                            int srcOffset = oc * outL;

                            for (int ol = 0; ol < outL; ol++)
                            {
                                output[b, oc, ol] = Unsafe.Add(ref pOut2D, srcOffset + ol) + bVal;
                            }
                        }
                    });
                }
            }

            return output;
        }

        /// <summary>
        /// Performs 2D Transposed Convolution (Deconvolution):
        /// Input [N, C_in, H, W] * Weight [C_in, C_out, K_h, K_w] -> Output [N, C_out, H_out, W_out].
        /// Essential for Diffusion VAE Latent Decoders and generative upsampling networks.
        /// </summary>
        public static Tensor<float> ConvTranspose2D(
            Tensor<float> input,
            Tensor<float> weight,
            Tensor<float>? bias = null,
            int stride = 1,
            int padding = 0,
            int outputPadding = 0,
            int dilation = 1)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (weight == null) throw new ArgumentNullException(nameof(weight));
            if (input.Rank != 4) throw new ArgumentException("Input tensor must have rank 4 [N, C_in, H, W].", nameof(input));
            if (weight.Rank != 4) throw new ArgumentException("Weight tensor must have rank 4 [C_in, C_out, K_h, K_w].", nameof(weight));

            int batchSize = input.Shape[0];
            int inChannels = input.Shape[1];
            int inH = input.Shape[2];
            int inW = input.Shape[3];

            int inChannelsW = weight.Shape[0];
            int outChannels = weight.Shape[1];
            int kernelH = weight.Shape[2];
            int kernelW = weight.Shape[3];

            if (inChannels != inChannelsW)
                throw new InvalidOperationException($"Channel mismatch: Input C_in={inChannels}, Weight C_in={inChannelsW}.");

            int outH = (inH - 1) * stride - 2 * padding + dilation * (kernelH - 1) + outputPadding + 1;
            int outW = (inW - 1) * stride - 2 * padding + dilation * (kernelW - 1) + outputPadding + 1;

            if (outH <= 0 || outW <= 0)
                throw new InvalidOperationException($"Computed output dimensions [{outH}x{outW}] must be positive.");

            var output = new Tensor<float>(batchSize, outChannels, outH, outW);
            var inContig = input.ToContiguous();
            var wContig = weight.ToContiguous();

            int inSpatial = inH * inW;
            int outSpatial = outH * outW;
            int kernelSpatial = kernelH * kernelW;

            unsafe
            {
                fixed (float* pIn = &inContig.Storage.GetPinnableReference(inContig.Offset))
                fixed (float* pW = &wContig.Storage.GetPinnableReference(wContig.Offset))
                fixed (float* pOut = &output.Storage.GetPinnableReference(output.Offset))
                {
                    IntPtr ptrIn = (IntPtr)pIn;
                    IntPtr ptrW = (IntPtr)pW;
                    IntPtr ptrOut = (IntPtr)pOut;

                    Parallel.For(0, batchSize * outChannels, bo =>
                    {
                        float* localIn = (float*)ptrIn;
                        float* localW = (float*)ptrW;
                        float* localOut = (float*)ptrOut;

                        int b = bo / outChannels;
                        int oc = bo % outChannels;

                        int outBase = b * outChannels * outSpatial + oc * outSpatial;
                        float bVal = bias != null ? bias[oc] : 0f;

                        // Initialize with bias
                        for (int i = 0; i < outSpatial; i++)
                        {
                            localOut[outBase + i] = bVal;
                        }

                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            int inBase = b * inChannels * inSpatial + ic * inSpatial;
                            int wBase = ic * outChannels * kernelSpatial + oc * kernelSpatial;

                            for (int ih = 0; ih < inH; ih++)
                            {
                                for (int iw = 0; iw < inW; iw++)
                                {
                                    float inVal = localIn[inBase + ih * inW + iw];
                                    if (inVal == 0f) continue;

                                    for (int kh = 0; kh < kernelH; kh++)
                                    {
                                        int oh = ih * stride - padding + kh * dilation;
                                        if (oh < 0 || oh >= outH) continue;

                                        int outRow = outBase + oh * outW;
                                        int wRow = wBase + kh * kernelW;

                                        for (int kw = 0; kw < kernelW; kw++)
                                        {
                                            int ow = iw * stride - padding + kw * dilation;
                                            if (ow >= 0 && ow < outW)
                                            {
                                                float wVal = localW[wRow + kw];
                                                localOut[outRow + ow] += inVal * wVal;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
            }

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

        /// <summary>
        /// Tiled Online Softmax FlashAttention on CPU with O(1) memory footprint.
        /// </summary>
        public static Tensor<float> FlashAttentionCpu(
            Tensor<float> q,
            Tensor<float> k,
            Tensor<float> v,
            Tensor<float>? mask = null,
            float? scale = null) =>
            TensorOps.FlashAttentionCpu(q, k, v, mask, scale);

        /// <summary>
        /// 1D Convolution over sequential or audio signals.
        /// </summary>
        public static Tensor<float> Conv1D(
            Tensor<float> input,
            Tensor<float> weight,
            Tensor<float>? bias = null,
            int stride = 1,
            int padding = 0,
            int dilation = 1) =>
            TensorOps.Conv1D(input, weight, bias, stride, padding, dilation);

        /// <summary>
        /// 2D Transposed Convolution (Deconvolution) for generative upsampling and Latent VAE Decoders.
        /// </summary>
        public static Tensor<float> ConvTranspose2D(
            Tensor<float> input,
            Tensor<float> weight,
            Tensor<float>? bias = null,
            int stride = 1,
            int padding = 0,
            int outputPadding = 0,
            int dilation = 1) =>
            TensorOps.ConvTranspose2D(input, weight, bias, stride, padding, outputPadding, dilation);

        /// <summary>
        /// Fused SwiGLU: y = SiLU(gate) * up.
        /// </summary>
        public static Tensor<float> SwiGLU(Tensor<float> gate, Tensor<float> up) =>
            TensorOps.SwiGLU(gate, up);

        /// <summary>
        /// Single-tensor Fused SwiGLU: splits tensor along dimension into [gate, up] and computes SwiGLU(gate, up).
        /// </summary>
        public static Tensor<float> SwiGLU(Tensor<float> x, int dim = -1) =>
            TensorOps.SwiGLU(x, dim);

        /// <summary>
        /// Fused GeGLU: y = GELU(gate) * up.
        /// </summary>
        public static Tensor<float> GeGLU(Tensor<float> gate, Tensor<float> up) =>
            TensorOps.GeGLU(gate, up);

        /// <summary>
        /// Single-tensor Fused GeGLU: splits tensor along dimension into [gate, up] and computes GeGLU(gate, up).
        /// </summary>
        public static Tensor<float> GeGLU(Tensor<float> x, int dim = -1) =>
            TensorOps.GeGLU(x, dim);

        /// <summary>
        /// Applies Rotary Positional Embedding (RoPE) in-place directly on the tensor.
        /// </summary>
        public static void ApplyRoPEInPlace(
            Tensor<float> x,
            int startPos = 0,
            float thetaBase = 10000.0f,
            bool interleaved = false) =>
            TensorOps.ApplyRoPEInPlace(x, startPos, thetaBase, interleaved);
    }
}
