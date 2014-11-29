using Neural_Network;

namespace NeuralNetworkTutorialApp
{
    using System;
    using System.Diagnostics;

    class Program
    {
        static void Main(string[] args)
        {
            var layerSizes = new int[3] {1, 2, 1};

            var transferFunctions = new TransferFunction[3]
            {TransferFunction.None, TransferFunction.Sigmoid, TransferFunction.Sigmoid};

            var backPropagationNetwork = new BackPropagationNetwork(layerSizes, transferFunctions);

            var input = new double[1] {0};
            var desired = new double[1] {0.13267};

            var output = new double[1];

            double error = 0.0;

            Stopwatch watch = Stopwatch.StartNew();

            for (int i = 0; i < 10001; i++)
            {
                error = backPropagationNetwork.Train(ref input, ref desired, 0.15, 0.1);

                backPropagationNetwork.Run(ref input, out output);

                if (i%100 == 0)
                {
                    Console.WriteLine("Iteration {0}:\n\tInput {1:0.00000} Output {2:0.00000} Error {3: 0.00000}", i, input[0],
                        output[0], error);
                }
            }

            watch.Stop();

            Console.WriteLine("Time Elapsed :" + watch.Elapsed);

            Console.ReadLine();
        }
    }
}
