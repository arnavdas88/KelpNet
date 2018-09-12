﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using Cloo;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Common.Functions
{
    [Serializable]
    public abstract class CompressibleFunction : SingleInputFunction, IParallelizable
    {
        const string FUNCTION_NAME = "CompressibleFunction";

        public CompressibleActivation Activator { get; protected set; }

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardgWKernel;

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardgXKernel;

        public string ForwardKernelName { get; }
        public string BackwardgWKernelName { get; }
        public string BackwardgXKernelName { get; }

        protected string KernelString;

        private readonly KeyValuePair<string, string>[] _activationParameters;

        protected abstract NdArray NeedPreviousForwardCpu(NdArray input);
        protected abstract NdArray NeedPreviousForwardGpu(NdArray input);
        protected abstract void NeedPreviousBackwardCpu(NdArray y, NdArray x);
        protected abstract void NeedPreviousBackwardGpu(NdArray y, NdArray x);

        protected CompressibleFunction(string functionName, CompressibleActivation activation = null, KeyValuePair<string, string>[] activationParameters = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            string kernelNameBase = functionName.Replace(" ", "");
            this.ForwardKernelName = kernelNameBase + "Forward";
            this.BackwardgWKernelName = kernelNameBase + "gWBackward";
            this.BackwardgXKernelName = kernelNameBase + "gXBackward";

            this.KernelString = Weaver.GetKernelSource(functionName);

            this._activationParameters = activationParameters;

            this.SetActivation(activation);

            this.SetGpuEnable(gpuEnable);
        }

        public bool SetGpuEnable(bool enable)
        {
            this.GpuEnable = enable & Weaver.Enable;

            this.CreateKernel();

            if (this.GpuEnable)
            {
                this.SingleInputForward = this.NeedPreviousForwardGpu;
                this.SingleOutputBackward = this.NeedPreviousBackwardGpu;
                foreach (var item in arrays)
                {
                    item.Value.Switch(ComputeDeviceTypes.Gpu);
                }
            }
            else
            {
                this.SingleInputForward = this.NeedPreviousForwardCpu;
                this.SingleOutputBackward = this.NeedPreviousBackwardCpu;
                foreach (var item in arrays)
                {
                    item.Value.Switch(ComputeDeviceTypes.Cpu);
                }
            }

            return GpuEnable;
        }

        //後からActivationを追加する用
        public void SetActivation(CompressibleActivation activation)
        {
            this.Activator = activation;

            if (this.Activator != null)
            {
                foreach (var activationParameterer in this._activationParameters)
                {
                    this.KernelString = this.KernelString.Replace(activationParameterer.Key, activationParameterer.Value);
                }
            }

            //Kernelの再構築が必要
            CreateKernel();
        }

        public void CreateKernel()
        {
            if (this.GpuEnable)
            {
                string kernelSource = this.KernelString;

                if (this.Activator != null)
                {
                    kernelSource = this.Activator.ActivateFunctionString + this.KernelString;
                }

                ComputeProgram program = Weaver.CreateProgram(kernelSource);
                this.ForwardKernel = program.CreateKernel(this.ForwardKernelName);
                this.BackwardgWKernel = program.CreateKernel(this.BackwardgWKernelName);
                this.BackwardgXKernel = program.CreateKernel(this.BackwardgXKernelName);
            }
        }

        Dictionary<string, RealArray> arrays = new Dictionary<string, RealArray>();
        protected RealArray CreateArray(string name, int len, ComputeMemoryFlags? memoryFlag = null)
        {
            arrays[name] = new RealArray(len);
            if (memoryFlag != null)
                arrays[name].GpuMemoryFlag = (ComputeMemoryFlags)memoryFlag;
            return arrays[name];
        }

        protected NdArray SetArray(string name, NdArray arr)
        {
            SetArray(name + ".Grad", arr.Grad);
            SetArray(name + ".Data", arr.Data);

            return arr;
        }

        protected RealArray SetArray(string name, RealArray arr)
        {
            arrays[name] = arr;
            return arr;
        }

        protected RealArray GetArray(string name)
        {
            return arrays[name];
        }

        protected RealArray GetArray(string name, int len, ComputeMemoryFlags? memoryFlag = null)
        {
            if (arrays.ContainsKey(name))
                return arrays[name];
            return CreateArray(name, len, memoryFlag);
        }
    }
}
