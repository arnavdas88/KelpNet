using Cloo;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
//using RealType = System.Double;
using RealType = System.Single;

namespace KelpNet.Common
{
    class RealTool
    {
        [DllImport("kernel32.dll")]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, int count);
    }

    [Serializable]
    public struct Real : IComparable<Real>
    {
        public readonly RealType Value;

        public static Int32 Size => sizeof(RealType);

        private Real(double value)
        {
            this.Value = (RealType)value;
        }

        public static implicit operator Real(double value)
        {
            return new Real(value);
        }

        public static implicit operator RealType(Real real)
        {
            return real.Value;
        }

        public int CompareTo(Real other)
        {
            return this.Value.CompareTo(other.Value);
        }

        public override string ToString()
        {
            return this.Value.ToString();
        }

        public static Real[] GetArray(Array data)
        {
            Type arrayType = data.GetType().GetElementType();
            Real[] resultData = new Real[data.Length];

            //型の不一致をここで吸収
            if (arrayType != typeof(RealType) && arrayType != typeof(Real))
            {
                //一次元の長さの配列を用意
                Array array = Array.CreateInstance(arrayType, data.Length);
                //一次元化して
                Buffer.BlockCopy(data, 0, array, 0, Marshal.SizeOf(arrayType) * resultData.Length);

                data = new RealType[array.Length];

                //型変換しつつコピー
                Array.Copy(array, data, array.Length);
            }

            //データを叩き込む
            int size = sizeof(RealType) * data.Length;
            GCHandle gchObj = GCHandle.Alloc(data, GCHandleType.Pinned);
            GCHandle gchBytes = GCHandle.Alloc(resultData, GCHandleType.Pinned);
            RealTool.CopyMemory(gchBytes.AddrOfPinnedObject(), gchObj.AddrOfPinnedObject(), size);
            gchObj.Free();
            gchBytes.Free();

            return resultData;
        }
    }

    public class RealArray : IDisposable, IEnumerable, IEnumerable<Real>
    {
        public ComputeDeviceTypes Device { get; set; }
        public ComputeMemoryFlags GpuMemoryFlag { get; set; } = ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer;
        public ComputeContext GpuComputeContext { get; } = Weaver.Context;
        public bool IsGpu { get; set; }
        public bool IsDisposed { get; set; } = false;

        public int Length => IsGpu ? (int)gpuData.Count : cpuData.Length;
        public int Count => Length;

        public StackTrace StackTrace { get; private set; } = new StackTrace(true);

        Real[] cpuData;
        ComputeBuffer<Real> gpuData;

        public RealArray(int count)
        {
            Device = ComputeDeviceTypes.Cpu;
            cpuData = new Real[count];
            IsGpu = false;
        }

        public RealArray(Real[] data)
        {
            Device = ComputeDeviceTypes.Cpu;
            cpuData = data;
            IsGpu = false;
        }

        public Real this[int index]
        {
            get
            {
                switch (Device)
                {
                    case ComputeDeviceTypes.Default:
                    case ComputeDeviceTypes.Cpu:
                        return cpuData[index];
                    case ComputeDeviceTypes.Gpu:
                    case ComputeDeviceTypes.Accelerator:
                    case ComputeDeviceTypes.All:
                    default:
                        throw new NotImplementedException();
                }
            }
            set
            {
                switch (Device)
                {
                    case ComputeDeviceTypes.Default:
                    case ComputeDeviceTypes.Cpu:
                        cpuData[index] = value;
                        break;
                    case ComputeDeviceTypes.Gpu:
                    case ComputeDeviceTypes.Accelerator:
                    case ComputeDeviceTypes.All:
                    default:
                        throw new NotImplementedException();
                }
            }
        }

        public static implicit operator RealArray(Real[] d)  // implicit digit to byte conversion operator
        {
            return new RealArray(d);
        }

        public static implicit operator Real[] (RealArray d)  // implicit digit to byte conversion operator
        {
            if (d.Device != ComputeDeviceTypes.Cpu)
                throw new NotImplementedException();

            return d.GetArray();
        }

        public class DeviceSwitchHandler : IDisposable
        {
            public bool Flush { get; set; } = true;

            ComputeDeviceTypes type;
            RealArray obj;

            public DeviceSwitchHandler(RealArray obj, ComputeDeviceTypes type)
            {
                this.obj = obj;
                this.type = type;
            }

            public void Dispose()
            {
                obj.Switch(type, Flush);
            }
        }

        public DeviceSwitchHandler Switch(ComputeDeviceTypes type, bool flush = true)
        {
            if (Device == type)
                return new DeviceSwitchHandler(this, Device);

            switch (type)
            {
                case ComputeDeviceTypes.Default:
                case ComputeDeviceTypes.Cpu:
                    IsGpu = false;
                    if (cpuData == null || cpuData.Length != gpuData.Count)
                        cpuData = new Real[(int)gpuData.Count];
                    Weaver.CommandQueue.ReadFromBuffer(gpuData, ref cpuData, true, null);
                    if (flush)
                    {
                        gpuData.Dispose();
                        gpuData = null;
                    }
                    break;
                case ComputeDeviceTypes.Gpu:
                    IsGpu = true;
                    if (gpuData != null)
                    {
                        if (gpuData.Count != cpuData.Length)
                        {
                            gpuData.Dispose();
                            gpuData = new ComputeBuffer<Real>(GpuComputeContext, GpuMemoryFlag, cpuData);
                        }
                        else
                        {
                            Weaver.CommandQueue.WriteToBuffer(cpuData, gpuData, true, null);
                        }
                    }
                    else
                    {
                        gpuData = new ComputeBuffer<Real>(GpuComputeContext, GpuMemoryFlag, cpuData);
                    }
                    if (flush)
                        cpuData = null;
                    break;
                case ComputeDeviceTypes.Accelerator:
                case ComputeDeviceTypes.All:
                default:
                    break;
            }

            var preDevice = Device;
            Device = type;

            return new DeviceSwitchHandler(this, preDevice);
        }

        public ComputeBuffer<Real> GetBuffer()
        {
            if (Device == ComputeDeviceTypes.Gpu)
                return gpuData;

            throw new NotImplementedException();
        }

        public Real[] GetArray()
        {
            switch (Device)
            {
                case ComputeDeviceTypes.Default:
                case ComputeDeviceTypes.Cpu:
                    return cpuData;
                case ComputeDeviceTypes.Gpu:
                    var gpuBuf = new Real[gpuData.Count];
                    Weaver.CommandQueue.ReadFromBuffer(gpuData, ref gpuBuf, true, null);
                    return gpuBuf;
                case ComputeDeviceTypes.Accelerator:
                case ComputeDeviceTypes.All:
                default:
                    throw new NotImplementedException();
            }
        }

        public Real[] ToArray()
        {
            switch (Device)
            {
                case ComputeDeviceTypes.Default:
                case ComputeDeviceTypes.Cpu:
                    return (Real[])cpuData.Clone();
                case ComputeDeviceTypes.Gpu:
                    var gpuBuf = new Real[gpuData.Count];
                    Weaver.CommandQueue.ReadFromBuffer(gpuData, ref gpuBuf, true, null);
                    return gpuBuf;
                case ComputeDeviceTypes.Accelerator:
                case ComputeDeviceTypes.All:
                default:
                    throw new NotImplementedException();
            }
        }

        public void Write(Real[] Data)
        {
            if (Data.Length != Length)
                throw new IndexOutOfRangeException();

            switch (Device)
            {
                case ComputeDeviceTypes.Default:
                case ComputeDeviceTypes.Cpu:
                    Array.Copy(Data, cpuData, Length);
                    break;
                case ComputeDeviceTypes.Gpu:
                    Weaver.CommandQueue.WriteToBuffer(Data, gpuData, true, null);
                    break;
                case ComputeDeviceTypes.Accelerator:
                case ComputeDeviceTypes.All:
                default:
                    throw new NotImplementedException();
            }
        }

        ~RealArray()
        {
            if (!IsDisposed && Device == ComputeDeviceTypes.Gpu)
            {
                Console.WriteLine($"RealArray is not disposed properly.\n===STACKTRACE===\n{StackTrace}\n============");
            }
            Console.WriteLine("something is not good");
            Dispose();
        }

        public void Dispose()
        {
            if (IsDisposed)
                return;

            cpuData = null;

            gpuData?.Dispose();
            gpuData = null;

            IsDisposed = true;

            GC.SuppressFinalize(this);
        }

        public IEnumerator GetEnumerator()
        {
            return cpuData.GetEnumerator();
        }

        IEnumerator<Real> IEnumerable<Real>.GetEnumerator()
        {
            return (IEnumerator<Real>)cpuData.GetEnumerator();
        }
    }

}
