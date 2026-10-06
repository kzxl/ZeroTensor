using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text;
using ZeroTensor.Core.Storage;

namespace ZeroTensor.Core
{
    /// <summary>
    /// Supported GGML numerical tensor data types in GGUF format containers.
    /// </summary>
    public enum GgmlType : uint
    {
        F32 = 0,
        F16 = 1,
        Q4_0 = 2,
        Q4_1 = 3,
        Q5_0 = 6,
        Q5_1 = 7,
        Q8_0 = 8,
        Q8_1 = 9,
        Q2_K = 10,
        Q3_K = 11,
        Q4_K = 12,
        Q5_K = 13,
        Q6_K = 14,
        Q8_K = 15,
        I8 = 16,
        I16 = 17,
        I32 = 18,
        I64 = 19,
        F64 = 20,
        BF16 = 30
    }

    /// <summary>
    /// Metadata descriptor for a tensor stored within a GGUF container.
    /// </summary>
    public sealed class GgufTensorInfo
    {
        public string Name { get; set; } = "";
        public int[] Dimensions { get; set; } = Array.Empty<int>();
        public GgmlType Type { get; set; }
        public ulong RelativeOffset { get; set; }
        public long AbsoluteOffset { get; set; }
        public long ByteSize { get; set; }
    }

    /// <summary>
    /// High-level memory-mapped container for GGUF model files.
    /// Retains shared file mapping lifetime and offers instant access to model weights.
    /// </summary>
    public sealed class GgufArchive : IDisposable
    {
        private readonly MemoryMappedFile _mmf;
        private readonly SharedResourceHolder _lifetime;
        private readonly Dictionary<string, object> _metadata;
        private readonly List<GgufTensorInfo> _tensorInfos;
        private readonly Dictionary<string, Tensor<float>> _fp32Tensors;

        public IReadOnlyDictionary<string, object> Metadata => _metadata;
        public IReadOnlyList<GgufTensorInfo> TensorInfos => _tensorInfos;
        public IReadOnlyDictionary<string, Tensor<float>> Tensors => _fp32Tensors;

        internal GgufArchive(
            MemoryMappedFile mmf,
            SharedResourceHolder lifetime,
            Dictionary<string, object> metadata,
            List<GgufTensorInfo> tensorInfos,
            Dictionary<string, Tensor<float>> fp32Tensors)
        {
            _mmf = mmf;
            _lifetime = lifetime;
            _metadata = metadata;
            _tensorInfos = tensorInfos;
            _fp32Tensors = fp32Tensors;
        }

        public void Dispose()
        {
            foreach (var kvp in _fp32Tensors)
            {
                kvp.Value.Dispose();
            }
            _lifetime.Dispose();
        }
    }

    /// <summary>
    /// High-performance pure C# GGUF (GGML Universal Format) file reader.
    /// Enables zero-copy memory-mapped loading of modern open-weights LLMs (LLaMA, Qwen, Mistral).
    /// </summary>
    public static class GgufFile
    {
        private const uint GgufMagic = 0x46554747; // 'GGUF' in little-endian

        /// <summary>
        /// Opens a GGUF file with true zero-copy OS memory-mapping.
        /// FP32 tensors are directly accessible as <see cref="Tensor{float}"/> backed by OS page cache.
        /// </summary>
        public static GgufArchive OpenMemoryMapped(string filePath)
        {
            if (string.IsNullOrEmpty(filePath)) throw new ArgumentNullException(nameof(filePath));
            if (!File.Exists(filePath)) throw new FileNotFoundException("GGUF file not found.", filePath);

            var metadata = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
            var tensorInfos = new List<GgufTensorInfo>();
            long dataBaseOffset;
            uint alignment = 32;

            // 1. Parse header and metadata from file stream
            using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var reader = new BinaryReader(fs, Encoding.UTF8))
            {
                uint magic = reader.ReadUInt32();
                if (magic != GgufMagic)
                {
                    throw new InvalidDataException($"Invalid GGUF magic header: 0x{magic:X8}, expected 0x{GgufMagic:X8}.");
                }

                uint version = reader.ReadUInt32();
                if (version < 2 || version > 3)
                {
                    throw new NotSupportedException($"GGUF version {version} is not supported (expected 2 or 3).");
                }

                ulong tensorCount = reader.ReadUInt64();
                ulong metadataKvCount = reader.ReadUInt64();

                // Read Metadata Key-Values
                for (ulong i = 0; i < metadataKvCount; i++)
                {
                    string key = ReadGgufString(reader);
                    uint valType = reader.ReadUInt32();
                    object val = ReadGgufValue(reader, valType);
                    metadata[key] = val;

                    if (key.Equals("general.alignment", StringComparison.OrdinalIgnoreCase) && val is uint alignVal)
                    {
                        alignment = alignVal;
                    }
                }

                // Read Tensor Info structures
                for (ulong i = 0; i < tensorCount; i++)
                {
                    string name = ReadGgufString(reader);
                    uint nDims = reader.ReadUInt32();

                    // GGUF stores dimensions in reverse (Fortran / col-major order)
                    var rawDims = new ulong[nDims];
                    for (uint d = 0; d < nDims; d++) rawDims[d] = reader.ReadUInt64();

                    var csharpDims = new int[nDims];
                    for (int d = 0; d < (int)nDims; d++)
                    {
                        csharpDims[d] = (int)rawDims[nDims - 1 - (uint)d];
                    }

                    uint ggmlType = reader.ReadUInt32();
                    ulong offset = reader.ReadUInt64();

                    tensorInfos.Add(new GgufTensorInfo
                    {
                        Name = name,
                        Dimensions = csharpDims,
                        Type = (GgmlType)ggmlType,
                        RelativeOffset = offset
                    });
                }

                // Compute aligned data start offset
                long currentPos = fs.Position;
                long remainder = currentPos % alignment;
                dataBaseOffset = remainder == 0 ? currentPos : currentPos + (alignment - remainder);
            }

            // 2. Open MemoryMappedFile and create zero-copy tensors
            var mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            var lifetime = new SharedResourceHolder(mmf, Math.Max(tensorInfos.Count, 1));
            var fp32Tensors = new Dictionary<string, Tensor<float>>();

            foreach (var info in tensorInfos)
            {
                info.AbsoluteOffset = dataBaseOffset + (long)info.RelativeOffset;
                var shape = new TensorShape(info.Dimensions);
                int elemCount = shape.TotalElements;

                if (info.Type == GgmlType.F32)
                {
                    info.ByteSize = (long)elemCount * sizeof(float);
                    var storage = new MemoryMappedStorage<float>(mmf, info.AbsoluteOffset, elemCount, ownsMmf: false, sharedLifetime: lifetime);
                    var strides = TensorStrides.ComputeContiguousStrides(shape);
                    fp32Tensors[info.Name] = new Tensor<float>(storage, 0, shape, strides);
                }
            }

            return new GgufArchive(mmf, lifetime, metadata, tensorInfos, fp32Tensors);
        }

        #region Helpers

        private static string ReadGgufString(BinaryReader reader)
        {
            ulong len = reader.ReadUInt64();
            if (len > int.MaxValue) throw new InvalidDataException("String length exceeds maximum supported size.");
            byte[] bytes = reader.ReadBytes((int)len);
            return Encoding.UTF8.GetString(bytes);
        }

        private static object ReadGgufValue(BinaryReader reader, uint type)
        {
            switch (type)
            {
                case 0: return reader.ReadByte();
                case 1: return reader.ReadSByte();
                case 2: return reader.ReadUInt16();
                case 3: return reader.ReadInt16();
                case 4: return reader.ReadUInt32();
                case 5: return reader.ReadInt32();
                case 6: return reader.ReadSingle();
                case 7: return reader.ReadBoolean();
                case 8: return ReadGgufString(reader);
                case 9: // Array
                    uint itemType = reader.ReadUInt32();
                    ulong count = reader.ReadUInt64();
                    var list = new object[count];
                    for (ulong i = 0; i < count; i++)
                    {
                        list[i] = ReadGgufValue(reader, itemType);
                    }
                    return list;
                case 10: return reader.ReadUInt64();
                case 11: return reader.ReadInt64();
                case 12: return reader.ReadDouble();
                default:
                    throw new NotSupportedException($"Unsupported GGUF metadata value type: {type}.");
            }
        }

        #endregion
    }
}
