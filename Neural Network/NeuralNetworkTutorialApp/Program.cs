﻿using Neural_Network;

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
            {TransferFunction.None, TransferFunction.RationalSigmoid, TransferFunction.Linear};

            var backPropagationNetwork = new BackPropagationNetwork(layerSizes, transferFunctions);

            var input = new double[1] {1.0};
            var desired = new double[1] {2.5};

            var output = new double[1];

            double error = 0.0;

            Stopwatch watch = Stopwatch.StartNew();

            for (int i = 0; i < 2001; i++)
            {
                error = backPropagationNetwork.Train(ref input, ref desired, 0.15, 0.1);

                backPropagationNetwork.Run(ref input, out output);

                if (i%200 == 0)
                {
                    Console.WriteLine("Iteration {0}:\n\tInput {1:0.000} Output {2:0.000} Error {3: 0.000}", i, input[0],
                        output[0], error);
                }
            }

            watch.Stop();

            Console.WriteLine("Time Elapsed :" + watch.Elapsed);

            backPropagationNetwork.Save(@"C:\Users\Dan\Documents\GitHub\NeuralNetworkTutorial\Neural Network\NeuralNetworkTutorialApp\test_network.xml");

            var backPropagationNetwork2 = new BackPropagationNetwork(@"C:\Users\Dan\Documents\GitHub\NeuralNetworkTutorial\Neural Network\NeuralNetworkTutorialApp\test_network.xml");



            backPropagationNetwork.Run(ref input, out output);
            Console.WriteLine("Test: Input {0:0.000} Output {1:0.000}", input[0], output[0]);

            Console.ReadLine();
        }
    }
}
