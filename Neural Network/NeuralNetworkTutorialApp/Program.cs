using Neural_Network;

namespace NeuralNetworkTutorialApp
{
    using System;
    using System.Diagnostics;

    class Program
    {
        private const string _filePath = @"C:\Users\Dan\Documents\GitHub\NeuralNetworkTutorial\Neural Network\NeuralNetworkTutorialApp\test_network.xml";

        static void Main(string[] args)
        {
            var layerSizes = new[] {2, 2, 1};

            var transferFunctions = new[]
            {TransferFunction.None, TransferFunction.Sigmoid, TransferFunction.Linear};

            var backPropagationNetwork = new BackPropagationNetwork(layerSizes, transferFunctions)
            {
                Name = "XOR-Gate Example"
            };

            var input = new double[4][];

            var expected = new double[4][];

            for (int i = 0; i < 4; i++)
            {
                input[i] = new double[2];
                expected[i] = new double[1];
            }

            input[0][0] = 0.0;
            input[0][1] = 0.0;
            expected[0][0] = 0; // false xor false = false

            input[1][0] = 1.0;
            input[1][1] = 0.0;
            expected[1][0] = 1; // true xor false = true

            input[2][0] = 0.0;
            input[2][1] = 1.0;
            expected[2][0] = 1; // false xor true = true

            input[3][0] = 1.0;
            input[3][1] = 1.0;
            expected[3][0] = 0; // true xor true = false


            double error = 0.0;
            const int maxCount = 10;
            int count = 0;

            Stopwatch watch = Stopwatch.StartNew();

            do
            {
                // prepare for training epic
                count ++;
                error = 0;

                // train
                for (int i = 0; i < 4; i++)
                {
                    error += backPropagationNetwork.Train(ref input[i], ref expected[i], .15, .1);
                }

                if (count % 1 == 0)
                {
                    Console.WriteLine("Epoch {0} completed with error {1:0.0000}", count, error);
                }
            } while (error > 0.0001 && count <= maxCount);

            watch.Stop();

            var output = new double[4][];

            for (int i = 0; i < 4; i++)
            {
                backPropagationNetwork.Run(ref input[i], out output[i]);
            }

            for (int i = 0; i < 4; i++)
            {
                Console.WriteLine("For inputs {0} and {1}, output is {2}", input[i][0], input[i][1], output[i][0]);
            }

            Console.WriteLine("Time Elapsed :" + watch.Elapsed);
            Console.WriteLine("Hit Enter...");
            Console.ReadLine();
        }
    }
}
