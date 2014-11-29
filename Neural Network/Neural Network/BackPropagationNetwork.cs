namespace Neural_Network
{
    using System;

    public enum TransferFunction
    {
        None,
        Sigmoid,
        Linear,
        Gaussian,
        RationalSigmoid
    }

    static class TransferFunctions
    {
        public static double Evaluate(TransferFunction transferFunction, double input)
        {
            switch (transferFunction)
            {
                case TransferFunction.Sigmoid:
                    return Sigmoid(input);
                case TransferFunction.Linear:
                    return Linear(input);
                case TransferFunction.Gaussian:
                    return Gaussian(input);
                case TransferFunction.RationalSigmoid:
                    return RationalSigmoid(input);
                default:
                    return 0.0;
            }
        }

        public static double EvaluateDerivative(TransferFunction transferFunction, double input)
        {
            switch (transferFunction)
            {
                case TransferFunction.Sigmoid:
                    return SigmoidDerivative(input);
                case TransferFunction.Linear:
                    return LinearDerivative(input);
                case TransferFunction.Gaussian:
                    return GaussianDerivative(input);
                case TransferFunction.RationalSigmoid:
                    return RationalSigmoidDerivative(input);
                default:
                    return 0.0;
            }
        }

        /* Transfer Function definitions */

        // currently returns infinity - must fix equation or find out why x = 0
        private static double Sigmoid(double x)
        {
            var result = 1.0/(1.0 + Math.Exp(-x));

            return result;
        }

        private static double SigmoidDerivative(double x)
        {
            var result = Sigmoid(x)*(1 - Sigmoid(x));

            return result;
        }

        private static double Linear(double x)
        {
            return x;
        }

        private static double LinearDerivative(double x)
        {
            return 1;
        }

        private static double Gaussian(double x)
        {
            return Math.Exp(-Math.Pow(x, 2));
        }

        private static double GaussianDerivative(double x)
        {
            return (-2*x*Gaussian(x));
        }

        private static double RationalSigmoid(double x)
        {
            return (x / (1.0 + Math.Sqrt(1.0 + x * x)));
        }

        private static double RationalSigmoidDerivative(double x)
        {
            double val = Math.Sqrt(1 + x*x);

            return (1.0/val*(1 + val));
        }
    }

    public class BackPropagationNetwork
    {
        private int LayerCount { get; set; }

        private int InputSize { get; set; }

        private int[] LayerSize { get; set; }

        private TransferFunction[] TransferFunctions { get; set; }

        private double[][] LayerOutput { get; set; }

        private double[][] LayerInput { get; set; }

        private double[][] Bias { get; set; }

        private double[][] Delta { get; set; }

        private double[][] PreviousBiasDelta { get; set; }

        private double[][][] Weight { get; set; }

        private double[][][] PreviousWeightDelta { get; set; }

        public BackPropagationNetwork(int[] layerSizes, TransferFunction[] transferFunctions)
        {
            if(transferFunctions.Length != layerSizes.Length)
                throw new ArgumentException("There is not an equal number of layers and transfer functions.");
            if (transferFunctions[0] != TransferFunction.None)
                throw new ArgumentException("The first transfer function must be None");

            LayerCount = layerSizes.Length - 1;
            InputSize = layerSizes[0];
            LayerSize = new int[LayerCount];

            for (int i = 0; i < LayerCount; i++)
                LayerSize[i] = layerSizes[i + 1];

            TransferFunctions = new TransferFunction[LayerCount];

            for (int i = 0; i < LayerCount; i++)
                TransferFunctions[i] = transferFunctions[i + 1];

            Bias = new double[LayerCount][];
            PreviousBiasDelta = new double[LayerCount][];
            Delta = new double[LayerCount][];
            LayerOutput = new double[LayerCount][];
            LayerInput = new double[LayerCount][];
            Weight = new double[LayerCount][][];
            PreviousWeightDelta = new double[LayerCount][][];

            for (int l = 0; l < LayerCount; l++)
            {
                Bias[l] = new double[LayerSize[l]];
                PreviousBiasDelta[l] = new double[LayerSize[l]];
                Delta[l] = new double[LayerSize[l]];
                LayerOutput[l] = new double[LayerSize[l]];
                LayerInput[l] = new double[LayerSize[l]];

                Weight[l] = new double[l == 0 ? InputSize : LayerSize[l-1]][];
                PreviousWeightDelta[l] = new double[l == 0 ? InputSize : LayerSize[l-1]][];

                for (int i = 0; i < (l == 0 ? InputSize : LayerSize[l - 1]); i++)
                {
                    Weight[l][i] = new double[LayerSize[l]];
                    PreviousWeightDelta[l][i] = new double[LayerSize[l]];
                }
            }
        }

        public void Run(ref double[] inputValues, out double[] output)
        {
            if (inputValues.Length != InputSize)
                throw new ArgumentException("Input Data is not of the correct dimension.");

            output = new double[LayerSize[LayerCount-1]];

            for (int l = 0; l < LayerCount; l++)
            {
                for (int j = 0; j < LayerSize[l]; j++)
                {
                    double sum = 0.0;

                    for (int i = 0; i < (l == 0 ? InputSize : LayerSize[l - 1]); i++)
                    {
                        sum += Weight[l][i][j]*(l == 0 ? inputValues[i] : LayerOutput[l - 1][i]);

                        sum += Bias[l][j];
                        LayerInput[l][j] = sum;

                        LayerOutput[l][j] = Neural_Network.TransferFunctions.Evaluate(TransferFunctions[l], sum);
                    }
                }
            }

            for (int i = 0; i < LayerSize[LayerCount - 1]; i++)
            {
                output[i] = LayerOutput[LayerCount - 1][i];
            }
        }

        public double Train(ref double[] input, ref double[] desired, double trainingRate, double momentum)
        {
            if (input.Length != InputSize)
                throw new ArgumentException("The input is of the wrong dimension.", "input");
            if (desired.Length != LayerSize[LayerCount - 1])
                throw new ArgumentException("The input is of the wrong dimension.", "desired");

            double error = 0.0;
            double sum = 0.0;
            double weightDelta = 0.0;
            double biasDelta = 0.0;
            var output = new double[LayerSize[LayerCount - 1]];

            Run(ref input, out output);

            // do the back propagation
            for (int l = LayerCount - 1; l >= 0; l--)
            {
                if (l == LayerCount - 1) // in the output layer
                {
                    for (int k = 0; k < LayerSize[l]; k++)
                    {
                        Delta[l][k] = output[k] - desired[k];

                        error += Math.Pow(Delta[l][k], 2);

                        Delta[l][k] *= Neural_Network.TransferFunctions.EvaluateDerivative(TransferFunctions[l],
                            LayerInput[l][k]);
                    }
                }
                else // in a hidden layer
                {
                    for (int i = 0; i < LayerSize[l]; i++)
                    {
                        sum = 0.0;
                        for (int j = 0; j < LayerSize[l + 1]; j++)
                        {
                            sum += Weight[l + 1][i][j] - Delta[i][j];
                        }

                        sum *= Neural_Network.TransferFunctions.EvaluateDerivative(TransferFunctions[l],
                            LayerInput[l][i]);

                        Delta[l][i] = sum;
                    }
                }
            }

            // update the weights and biases

            for (int l = 0; l < LayerCount; l++)
                for (int i = 0; i < (l == 0 ? InputSize : LayerSize[l - 1]); i++)
                {
                    for (int j = 0; j < LayerSize[l]; j++)
                    {
                        weightDelta = trainingRate*Delta[l][j]*(l == 0 ? input[i] : LayerOutput[l - 1][i]);
                        Weight[l][i][j] -= weightDelta + momentum*PreviousWeightDelta[l][i][j];

                        PreviousWeightDelta[l][i][j] = weightDelta + momentum * PreviousWeightDelta[l][i][j];
                    }
                }

            for (int l = 0; l < LayerCount; l++)
            {
                for (int i = 0; i < LayerSize[l]; i++)
                {
                    biasDelta = trainingRate*Delta[l][i];
                    Bias[l][i] -= biasDelta + momentum*PreviousBiasDelta[l][i];

                    PreviousBiasDelta[l][i] = biasDelta;
                }
            }

            return error;
        }
    }
}
