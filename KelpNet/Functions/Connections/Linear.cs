using System;
using System.Collections.Generic;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Linear : CompressibleFunction
    {
        const string FUNCTION_NAME = "Linear";

        private const string PARAM_NAME = "/*ForwardActivate*/";
        private const string PARAM_VALUE = "gpuYSum = ForwardActivate(gpuYSum);";

        public NdArray Weight;
        public NdArray Bias;

        public readonly bool NoBias;

        public readonly int InputCount;
        public readonly int OutputCount;

        public Linear(
            int inputCount,
            int outputCount,
            bool noBias = false,
            Array initialW = null,
            Array initialb = null,
            CompressibleActivation activation = null,
            string name = FUNCTION_NAME,
            string[] inputNames = null,
            string[] outputNames = null,
            bool gpuEnable = false
            ) : base(
                FUNCTION_NAME,
                activation,
                new[] { new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE) },
                name,
                inputNames,
                outputNames,
                gpuEnable)
        {
            OutputCount = outputCount;
            InputCount = inputCount;

            Weight = new NdArray(outputCount, inputCount);
            Weight.Name = Name + ".Weight";

            NoBias = noBias;

            Parameters = new NdArray[noBias ? 1 : 2];

            if (initialW == null)
            {
                Initializer.InitWeight(Weight);
            }
            else
            {
                Weight.Data = Real.GetArray(initialW);
            }

            Parameters[0] = Weight;

            if (!noBias)
            {
                Bias = new NdArray(outputCount);
                Bias.Name = Name + ".Bias";

                if (initialb != null)
                {
                    Bias.Data = Real.GetArray(initialb);
                }

                Parameters[1] = Bias;
            }

            SetArray(Weight.Name, Weight);
            SetArray(Bias.Name, Bias);
        }

        Real[] GetBiasedValue(int batchCount)
        {
            Real[] y = new Real[OutputCount * batchCount];

            for (int i = 0; i < batchCount; i++)
            {
                Array.Copy(Bias.Data.GetArray(), 0, y, i * OutputCount, Bias.Data.Length);
            }

            return y;
        }

        protected override NdArray NeedPreviousForwardCpu(NdArray x)
        {
            x.Data.Switch(Common.ComputeDeviceTypes.Cpu);
            Weight.Data.Switch(Common.ComputeDeviceTypes.Cpu);


            Real[] y = NoBias ? new Real[OutputCount * x.BatchCount] : GetBiasedValue(x.BatchCount);

            for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
            {
                for (int i = 0; i < OutputCount; i++)
                {
                    for (int j = 0; j < InputCount; j++)
                    {
                        y[batchCount * OutputCount + i] += x.Data[batchCount * InputCount + j] * Weight.Data[i * InputCount + j];
                    }
                }
            }

            if (Activator != null)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    y[i] = Activator.ForwardActivate(y[i]);
                }
            }

            return new NdArray(y, new RealArray(y.Length), new[] { OutputCount }, x.BatchCount, this);
        }

        protected override NdArray NeedPreviousForwardGpu(NdArray x)
        {
            var ytemp = NoBias ? new Real[OutputCount * x.BatchCount] : GetBiasedValue(x.BatchCount);
            var y = GetArray("y", ytemp.Length, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer);
            y.Write(ytemp);

            x.Data.Switch(Common.ComputeDeviceTypes.Gpu);
            Weight.Data.Switch(Common.ComputeDeviceTypes.Gpu);
            y.Switch(Common.ComputeDeviceTypes.Gpu);

            var gpuX = x.Data.GetBuffer();
            var gpuW = Weight.Data.GetBuffer();
            var gpuY = y.GetBuffer();

            ForwardKernel.SetMemoryArgument(0, gpuX);
            ForwardKernel.SetMemoryArgument(1, gpuW);
            ForwardKernel.SetMemoryArgument(2, gpuY);
            ForwardKernel.SetValueArgument(3, OutputCount);
            ForwardKernel.SetValueArgument(4, InputCount);

            Weaver.CommandQueue.Execute
                (
                    ForwardKernel,
                    null,
                    new long[] { OutputCount, x.BatchCount },
                    null,
                    null
                );

            Weaver.CommandQueue.Flush();
            Weaver.CommandQueue.Finish();

            return new NdArray(y, GetArray("y.Grad", y.Length, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer), new[] { OutputCount }, x.BatchCount, this);
        }

        RealArray GetActivatedgy(NdArray y)
        {
            Real[] activatedgY = new Real[y.Grad.Length];

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    int index = batchCount * this.OutputCount + i;
                    activatedgY[index] = this.Activator.BackwardActivate(y.Grad[index], y.Data[index]);
                }
            }

            return activatedgY;
        }

        void CalcBiasGrad(RealArray gy, int batchCount)
        {
            using (gy.Switch(Common.ComputeDeviceTypes.Cpu))
            using (Bias.Grad.Switch(Common.ComputeDeviceTypes.Cpu))
            {
                for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
                {
                    for (int i = 0; i < OutputCount; i++)
                    {
                        Bias.Grad[i] += gy[batchCounter * OutputCount + i];
                    }
                }
            }
        }

        protected override void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            RealArray activatedgy = Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.BatchCount);

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    Real gyData = activatedgy[i + batchCount * this.OutputCount];

                    for (int j = 0; j < this.InputCount; j++)
                    {
                        this.Weight.Grad[i * this.InputCount + j] += x.Data[j + batchCount * this.InputCount] * gyData;
                        x.Grad[j + batchCount * this.InputCount] += this.Weight.Data[i * this.InputCount + j] * gyData;
                    }
                }
            }
        }

        protected override void NeedPreviousBackwardGpu(NdArray y, NdArray x)
        {
            //Real[] gx = new Real[x.Data.Length];
            var gx = GetArray("gx", x.Data.Length);
            var activatedgy = Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.BatchCount);

            //using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, activatedgy))
            //{

            Weight.Grad.Switch(Common.ComputeDeviceTypes.Gpu);
            x.Data.Switch(Common.ComputeDeviceTypes.Gpu);
            activatedgy.Switch(Common.ComputeDeviceTypes.Gpu);

            var gpugW = Weight.Grad.GetBuffer();
            var gpuX = x.Data.GetBuffer();
            var gpugY = activatedgy.GetBuffer();

            //using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, this.Weight.Grad))
            //using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
            //{

            BackwardgWKernel.SetMemoryArgument(0, gpugY);
            BackwardgWKernel.SetMemoryArgument(1, gpuX);
            BackwardgWKernel.SetMemoryArgument(2, gpugW);
            BackwardgWKernel.SetValueArgument(3, y.BatchCount);
            BackwardgWKernel.SetValueArgument(4, OutputCount);
            BackwardgWKernel.SetValueArgument(5, InputCount);

            Weaver.CommandQueue.Execute
            (
                BackwardgWKernel,
                null,
                new long[] { InputCount, OutputCount },
                null,
                null
            );

            Weaver.CommandQueue.Finish();

            //TODO
            //Weaver.CommandQueue.ReadFromBuffer(gpugW, ref this.Weight.Grad, true, null);
            //}

            //using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
            //using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, this.Weight.Data))
            //{
            gx.Switch(Common.ComputeDeviceTypes.Gpu);
            Weight.Data.Switch(Common.ComputeDeviceTypes.Gpu);

            var gpugX = gx.GetBuffer();
            var gpuW = Weight.Data.GetBuffer();

            BackwardgXKernel.SetMemoryArgument(0, gpugY);
            BackwardgXKernel.SetMemoryArgument(1, gpuW);
            BackwardgXKernel.SetMemoryArgument(2, gpugX);
            BackwardgXKernel.SetValueArgument(3, y.BatchCount);
            BackwardgXKernel.SetValueArgument(4, OutputCount);
            BackwardgXKernel.SetValueArgument(5, InputCount);

            Weaver.CommandQueue.Execute
            (
                BackwardgXKernel,
                null,
                new long[] { InputCount, y.BatchCount },
                null,
                null
            );

            Weaver.CommandQueue.Finish();
            //Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
            //}
            //}

            using (x.Grad.Switch(Common.ComputeDeviceTypes.Cpu, true))
            using (gx.Switch(Common.ComputeDeviceTypes.Cpu))
            {
                var xGrad = x.Grad.GetArray();
                var gxArray = gx.GetArray();

                for (int i = 0; i < x.Grad.Length; i++)
                {
                    xGrad[i] += gxArray[i];
                }
            }
        }

        public Convolution2D AsConvolution2D()
        {
            return new Convolution2D(this);
        }
    }
}
