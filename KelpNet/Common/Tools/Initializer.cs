using System;
using System.Threading.Tasks;

namespace KelpNet.Common.Tools
{
    class Initializer
    {
        //初期値が入力されなかった場合、この関数で初期化を行う
        public static void InitWeight(NdArray array, double masterScale = 1)
        {
            double localScale = 1 / Math.Sqrt(2);
            int fanIn = GetFans(array.Shape);
            double s = localScale * Math.Sqrt(2.0 / fanIn);
            var procCount = Environment.ProcessorCount;

            Parallel.For(0, procCount, (id) =>
            {
                var mother = new Mother();
                var scale = 0.05;
                var count = array.Data.Length;
                for (int i = id; i < count; i += procCount)
                {
                    mother.Sigma = scale;
                    array.Data[i] = mother.RandomNormal() * masterScale;
                }
            });

        }

        private static int GetFans(int[] shape)
        {
            int result = 1;

            for (int i = 1; i < shape.Length; i++)
            {
                result *= shape[i];
            }

            return result;
        }
    }
}
