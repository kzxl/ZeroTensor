using System;
using System.Collections.Generic;
using System.IO;
using Xunit;
using ZeroPrimitives.Memory;
using ZeroTensor.Core;
using ZeroTensor.Core.Storage;

namespace ZeroTensor.Tests
{
    public class TensorStorageAndSafetensorsMmapTests
    {
        [Fact]
        public void ManagedArrayStorage_BasicReadWrite()
        {
            using var storage = new ManagedArrayStorage<float>(8);
            Assert.Equal(8, storage.Length);
            Assert.Equal(DeviceType.Cpu, storage.Device);
            Assert.True(storage.IsCpuAccessible);
            Assert.False(storage.IsDisposed);

            var span = storage.AsSpan(0, 8);
            for (int i = 0; i < 8; i++) span[i] = i * 2.5f;

            var readSpan = storage.AsReadOnlySpan(0, 8);
            for (int i = 0; i < 8; i++) Assert.Equal(i * 2.5f, readSpan[i]);

            var arr = storage.TryGetArray(out int offset);
            Assert.NotNull(arr);
            Assert.Equal(0, offset);
        }

        [Fact]
        public void NativeMemoryStorage_AllocateAndWrapTensor()
        {
            using var storage = NativeMemoryStorage<float>.Allocate(12);
            Assert.Equal(12, storage.Length);
            Assert.Equal(DeviceType.Cpu, storage.Device);
            Assert.True(storage.IsCpuAccessible);
            Assert.False(storage.IsDisposed);

            var span = storage.AsSpan(0, 12);
            for (int i = 0; i < 12; i++) span[i] = (float)i;

            // Wrap into a 3x4 Tensor view
            var shape = new TensorShape(3, 4);
            var strides = TensorStrides.ComputeContiguousStrides(shape);
            var tensor = new Tensor<float>(storage, 0, shape, strides);

            Assert.Equal(3, tensor.Shape[0]);
            Assert.Equal(4, tensor.Shape[1]);
            Assert.Equal(5f, tensor[1, 1]);
            Assert.Equal(11f, tensor[2, 3]);

            // Arithmetic operation
            var doubled = tensor * 2f;
            Assert.Equal(10f, doubled[1, 1]);
            Assert.Equal(22f, doubled[2, 3]);
        }

        [Fact]
        public void FromNativeBlock_ZeroCopyExecution()
        {
            using var block = NativeMemoryBlock.Allocate(64 * sizeof(float));
            var tensor = Tensor.FromNativeBlock<float>(block, 8, 8);

            Assert.Equal(new[] { 8, 8 }, tensor.Shape.Dimensions);
            Assert.IsType<NativeMemoryStorage<float>>(tensor.Storage);

            tensor[3, 4] = 42f;
            Assert.Equal(42f, tensor[3, 4]);

            // SubTensor zero-copy slice
            var row3 = tensor.SubTensor(3);
            Assert.Equal(42f, row3[4]);
            row3[4] = 100f;
            Assert.Equal(100f, tensor[3, 4]);
        }

        [Fact]
        public void Safetensors_OpenMemoryMapped_TrueZeroCopy()
        {
            string tempFile = Path.Combine(Path.GetTempPath(), $"safetensors_test_{Guid.NewGuid():N}.safetensors");
            try
            {
                // 1. Create tensors and save to Safetensors file
                var w = Tensor.FromArray(new float[]
                {
                    1f, 2f,
                    3f, 4f
                }, 2, 2);

                var b = Tensor.FromArray(new float[] { 0.5f, -0.5f }, 2);

                var dict = new Dictionary<string, Tensor<float>>
                {
                    { "linear.weight", w },
                    { "linear.bias", b }
                };

                SafetensorsFile.Save(tempFile, dict);
                Assert.True(File.Exists(tempFile));

                // 2. Open via true zero-copy OS memory-mapping
                var loaded = SafetensorsFile.OpenMemoryMapped(tempFile);
                Assert.Equal(2, loaded.Count);

                var loadedW = loaded["linear.weight"];
                var loadedB = loaded["linear.bias"];

                // Verify backing storage is MemoryMappedStorage
                Assert.IsType<MemoryMappedStorage<float>>(loadedW.Storage);
                Assert.IsType<MemoryMappedStorage<float>>(loadedB.Storage);

                // Verify values
                Assert.Equal(1f, loadedW[0, 0]);
                Assert.Equal(2f, loadedW[0, 1]);
                Assert.Equal(3f, loadedW[1, 0]);
                Assert.Equal(4f, loadedW[1, 1]);
                Assert.Equal(0.5f, loadedB[0]);
                Assert.Equal(-0.5f, loadedB[1]);

                // 3. Compute inference using memory-mapped weights
                var x = Tensor.FromArray(new float[] { 10f, 20f }, 1, 2);
                // x @ W^T + b: [1, 2] @ [2, 2] = [1, 2]
                var y = TensorBlas.MatMul(x, loadedW) + loadedB;

                // y[0,0] = 10*1 + 20*3 + 0.5 = 70.5
                // y[0,1] = 10*2 + 20*4 - 0.5 = 99.5
                Assert.Equal(70.5f, y[0, 0]);
                Assert.Equal(99.5f, y[0, 1]);

                // Dispose tensors
                loadedW.Dispose();
                loadedB.Dispose();
            }
            finally
            {
                if (File.Exists(tempFile))
                {
                    try { File.Delete(tempFile); } catch { }
                }
            }
        }

        [Fact]
        public void BatchedMatMul_BroadcastingWithoutCopyTrap()
        {
            // A has shape [1, 2, 3] (e.g. weights shared across batch)
            var a = Tensor.FromArray(new float[]
            {
                1f, 2f, 3f,
                4f, 5f, 6f
            }, 1, 2, 3);

            // B has shape [4, 3, 2] (batch of 4 items)
            var bData = new float[4 * 3 * 2];
            for (int i = 0; i < bData.Length; i++) bData[i] = i + 1;
            var b = Tensor.FromArray(bData, 4, 3, 2);

            // Batched multiplication with broadcast: [1, 2, 3] x [4, 3, 2] -> [4, 2, 2]
            var c = TensorBlas.MatMul(a, b);

            Assert.Equal(new[] { 4, 2, 2 }, c.Shape.Dimensions);

            // Verify each batch slice independently
            var a2D = a.SubTensor(0); // [2, 3]
            for (int batch = 0; batch < 4; batch++)
            {
                var b2D = b.SubTensor(batch); // [3, 2]
                var expectedSlice = TensorBlas.MatMul2D(a2D, b2D); // [2, 2]

                var actualSlice = c.SubTensor(batch);
                for (int r = 0; r < 2; r++)
                {
                    for (int col = 0; col < 2; col++)
                    {
                        Assert.Equal(expectedSlice[r, col], actualSlice[r, col]);
                    }
                }
            }
        }
    }
}
