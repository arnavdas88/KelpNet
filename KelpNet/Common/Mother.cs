using System;

namespace KelpNet.Common
{
    //乱数の素
    //C#ではRandomを複数同時にインスタンスすると似たような値しか吐かないため
    //一箇所でまとめて管理しておく必要がある
    public class Mother
    {
        public static Mother Current = new Mother();
        public static Random Dice => Current.Rnd;

#if DEBUG
        public Random Rnd = new Random(128);
#else
        public Random Rnd = new Random();
#endif

        double Alpha, Beta, BoxMuller1, BoxMuller2;
        bool Flip;
        public double Mu = 0;
        public double Sigma = 1;

        // 平均mu, 標準偏差sigmaの正規分布乱数を得る。Box-Muller法による。
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        public double RandomNormal()
        {
            if (!Flip)
            {
                Alpha = Rnd.NextDouble();
                Beta = Rnd.NextDouble() * Math.PI * 2;
                BoxMuller1 = Math.Sqrt(-2 * Math.Log(Alpha));
                BoxMuller2 = Math.Sin(Beta);
            }
            else
            {
                BoxMuller2 = Math.Cos(Beta);
            }

            Flip = !Flip;

            return Sigma * (BoxMuller1 * BoxMuller2) + Mu;
        }
    }
}
