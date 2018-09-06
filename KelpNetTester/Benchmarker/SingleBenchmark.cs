using System;
using System.Diagnostics;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Noise;
using KelpNet.Functions.Poolings;

namespace KelpNetTester.Benchmarker
{
    class SingleBenchmark
    {
        const int INPUT_SIZE = 7 * 7 * 512;
        const int OUTPUT_SIZE = 4096;

        static Stopwatch sw;

        static double IsArrayEqual(Real[] a, Real[] b)
        {
            if (a.Length != b.Length)
                return double.MaxValue;

            double sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                var diff = Math.Abs(a[i] - b[i]);
                sum += diff;
            }

            return sum / a.Length;
        }

        public static void TestLayer(Function function, NdArray inputs)
        {
            double diff = 0;

            Console.WriteLine($"=== Test {function.Name} ===");

            sw.Restart();
            NdArray[] gradArrayCpu = function.Forward(inputs);
            sw.Stop();
            Console.WriteLine($"({function.Name}) Forward [Cpu] : {sw.Elapsed.TotalMilliseconds * 1000:n0}µs");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            function.Backward(inputs);
            sw.Stop();
            Console.WriteLine($"({function.Name}) Backward[Cpu] : {sw.Elapsed.TotalMilliseconds * 1000:n0}µs");

            if (function is CompressibleFunction compressible && compressible.SetGpuEnable(true))
            {
                var startMs = DateTime.Now.TimeOfDay.TotalMilliseconds;
                var fps = 0.0;
                var frame = 0;
                var testIndex = 0;
                var testCount = 10000;

                while (testIndex < testCount)
                {
                    var forwardHeader  = $"({function.Name}) Forward  [Gpu] {testIndex + 1:00}/{testCount} :";
                    var backwardHeader = $"({function.Name}) Backward [Gpu] {testIndex + 1:00}/{testCount} :";

                    sw.Restart();
                    NdArray[] gradArrayGpu = function.Forward(inputs);
                    sw.Stop();
                    Console.WriteLine($"{forwardHeader} {sw.Elapsed.TotalMilliseconds * 1000:n0}µs / fps : {fps}");

                    diff = IsArrayEqual(gradArrayCpu[0].Data.GetArray(), gradArrayGpu[0].Data.GetArray());
                    if (diff > 0.001)
                        Console.WriteLine($"{forwardHeader} GPU.Data != CPU.Data {diff}");

                    gradArrayGpu[0].Grad.Write(gradArrayGpu[0].Data.GetArray());

                    sw.Restart();
                    function.Backward(gradArrayGpu);
                    sw.Stop();
                    Console.WriteLine($"{backwardHeader} {sw.Elapsed.TotalMilliseconds * 1000:n0}µs / fps : {fps}");

                    var gpuData = gradArrayGpu[0].Data.GetArray();
                    var gpuGrad = gradArrayGpu[0].Grad.GetArray();

                    diff = IsArrayEqual(gradArrayCpu[0].Data.GetArray(), gpuData);
                    if (diff > 0.001)
                        Console.WriteLine($"{backwardHeader} GPU.Data != CPU.Data {diff}");
                    diff = IsArrayEqual(gradArrayCpu[0].Grad.GetArray(), gpuGrad);
                    if (diff > 0.001)
                        Console.WriteLine($"{backwardHeader} GPU.Data != CPU.Data {diff}");

                    frame++;
                    testIndex++;
                    if (DateTime.Now.TimeOfDay.TotalMilliseconds - startMs > 250)
                    {
                        fps = frame * 4;
                        startMs = DateTime.Now.TimeOfDay.TotalMilliseconds;
                        frame = 0;
                    }
                }
            }
        }

        public static void Run()
        {
            sw = new Stopwatch();

            Console.WriteLine("Generating Test Data...");
            NdArray input = new NdArray(BenchDataMaker.GetRealArray(INPUT_SIZE));
            NdArray inputImage = new NdArray(BenchDataMaker.GetRealArray(3 * 256 * 256 * 5), new[] { 3, 256, 256 }, 5);
            Console.WriteLine("Generated Test Data");

            Console.WriteLine("Init Linear");
            Linear linear = new Linear(INPUT_SIZE, OUTPUT_SIZE);

            Console.WriteLine("Init Tanh");
            Tanh tanh = new Tanh();
            Console.WriteLine("Init Sigmoid");
            Sigmoid sigmoid = new Sigmoid();
            Console.WriteLine("Init ReLU");
            ReLU relu = new ReLU();
            Console.WriteLine("Init LeakyReLU");
            LeakyReLU leakyRelu = new LeakyReLU();

            Console.WriteLine("Init MaxPooling");
            MaxPooling maxPooling = new MaxPooling(2);
            Console.WriteLine("Init Convolution2D");
            Convolution2D conv2d = new Convolution2D(3, 32, 3);
            Console.WriteLine("Init Deconvolution2D");
            Deconvolution2D deconv2d = new Deconvolution2D(32, 3, 3);

            Dropout dropout = new Dropout();

            TestLayer(linear, input);
            Console.WriteLine("aaaaaaaaaaaa");
            Console.ReadLine();

            TestLayer(tanh, input);
            TestLayer(sigmoid, input);
            TestLayer(relu, input);
            TestLayer(leakyRelu, input);

            TestLayer(maxPooling, inputImage);
            TestLayer(conv2d, inputImage);
            TestLayer(deconv2d, inputImage);

            TestLayer(dropout, input);
        }
    }
}
