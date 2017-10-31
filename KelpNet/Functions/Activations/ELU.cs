﻿using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ELU : SingleInputFunction
    {
        const string FUNCTION_NAME = "ELU";

        private readonly Real _alpha;

        public ELU(double alpha = 1, string name = FUNCTION_NAME) : base(name)
        {
            this._alpha = alpha;

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        private NdArray NeedPreviousForwardCpu(NdArray x)
        {
            Real[] result = new Real[x.Data.Length];

            for (int i = 0; i < x.Data.Length; i++)
            {
                if (x.Data[i] >= 0)
                {
                    result[i] = x.Data[i];
                }
                else
                {
                    result[i] = this._alpha * (Math.Exp(x.Data[i]) - 1);
                }
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, this);
        }

        private void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                if (x.Data[i] >= 0)
                {
                    x.Grad[i] += y.Grad[i];
                }
                else
                {
                    x.Grad[i] += y.Grad[i] * this._alpha * Math.Exp(x.Data[i]);
                }
            }
        }
    }
}
