//#define STRHC2
// STRHC2 = 10 gesture types = Use Data.dat
// No STRHC2 = 4 gesture types = Use Data1.dat

using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using Accord.Statistics.Kernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApplication1
{
    class Program
    {
        static void Main(string[] args)
        {
            string strFile;

#if STRHC2
            strFile = @".\Data.dat";
#else
            strFile = @".\Data1.dat";
#endif

            Dictionary<int, Tuple<Dictionary<string, List<List<double[]>>>, Dictionary<string, List<List<double[]>>>>> dict;
            using (System.IO.FileStream fs = System.IO.File.OpenRead(strFile))
            {
                BinaryFormatter bf = new BinaryFormatter();
                dict = bf.Deserialize(fs) as Dictionary<int, Tuple<Dictionary<string, List<List<double[]>>>, Dictionary<string, List<List<double[]>>>>>;
            }

            using (System.IO.StreamWriter sw = new System.IO.StreamWriter(@".\Output.tsv", false))
            {
                foreach (KeyValuePair<int, Tuple<Dictionary<string, List<List<double[]>>>, Dictionary<string, List<List<double[]>>>>> kvp in dict)
                {
                    Program.WriteStream(sw, string.Format("Fold {0}", kvp.Key));

                    int nFoldCorrect = 0, nFoldIncorrect = 0;

                    foreach (KeyValuePair<string, List<List<double[]>>> kvpTest in kvp.Value.Item2)
                    {
                        int nLabelCorrect = 0, nLabelIncorrect = 0;

                        foreach (List<double[]> gestTest in kvpTest.Value)
                        {
                            double[] gestTestConcat = Matrix.Concatenate(gestTest.ToArray());
                            double minCost = double.MaxValue;
                            string minLabel = string.Empty;

                            System.Diagnostics.Stopwatch watch = new System.Diagnostics.Stopwatch();
                            watch.Start();
                            foreach (KeyValuePair<string, List<List<double[]>>> kvpTrain in kvp.Value.Item1)
                            {
                                foreach (List<double[]> gestTrain in kvpTrain.Value)
                                {
                                    double[] gestTrainConcat = Matrix.Concatenate(gestTrain.ToArray());

                                    NDtw.Dtw dtw = new NDtw.Dtw(gestTestConcat, gestTrainConcat, NDtw.DistanceMeasure.Euclidean);
                                    double cost = dtw.GetCost();

                                    if (cost < minCost)
                                    {
                                        minCost = cost;
                                        minLabel = kvpTrain.Key;
                                    }
                                }
                            }
                            watch.Stop();

                            Program.WriteStream(sw, string.Format("Actual: {0}\tRecognized: {1}\tCost: {2}\tTime\t{3}", kvpTest.Key, minLabel, minCost, watch.ElapsedMilliseconds));

                            if (kvpTest.Key == minLabel)
                            {
                                nFoldCorrect++;
                                nLabelCorrect++;
                            }
                            else
                            {
                                nFoldIncorrect++;
                                nLabelIncorrect++;
                            }
                        }

                        Program.WriteStream(sw, string.Empty);
                        Program.WriteStream(sw, string.Format("{0}:\tCorrect = {1}\tIncorrect = {2}", kvpTest.Key, nLabelCorrect, nLabelIncorrect));
                    }

                    Program.WriteStream(sw, string.Empty);
                    Program.WriteStream(sw, string.Format("Fold {0}:\tCorrect = {1}\tIncorrect = {2}", kvp.Key, nFoldCorrect, nFoldIncorrect));
                    Program.WriteStream(sw, string.Empty);
                    Program.WriteStream(sw, string.Empty);
                    Program.WriteStream(sw, string.Empty);
                }
            }
        }

        static void WriteStream(System.IO.StreamWriter sw, string message)
        {
            Console.WriteLine(message);

            sw.WriteLine(message);
            sw.Flush();
        }

        static void Main_Orig(string[] args)
        {
            // I stopped because I just realized they only differentiate between TWO classes.  Instead, I applied Ndtw.

            //double[][][] sequences =
            //{
            //    new double[][] // first sequence
            //    {
            //        new double[] { 1, 1, 1 }, // first observation of the first sequence
            //        new double[] { 1, 2, 1 }, // second observation of the first sequence
            //        new double[] { 1, 4, 2 }, // third observation of the first sequence
            //        new double[] { 2, 2, 2 }, // fourth observation of the first sequence
            //    },

            //    new double[][] // second sequence (note that this sequence has a different length)
            //    {
            //        new double[] { 1, 1, 1 }, // first observation of the second sequence
            //        new double[] { 1, 5, 6 }, // second observation of the second sequence
            //        new double[] { 2, 7, 1 }, // third observation of the second sequence
            //    },

            //    new double[][] // third sequence 
            //    {
            //        new double[] { 8, 2, 1 }, // first observation of the third sequence
            //    },

            //    new double[][] // fourth sequence 
            //    {
            //        new double[] { 8, 2, 5 }, // first observation of the fourth sequence
            //        new double[] { 1, 5, 4 }, // second observation of the fourth sequence
            //    }
            //};

            //// Now, we will also have different class labels associated which each 
            //// sequence. We will assign -1 to sequences whose observations start 
            //// with { 1, 1, 1 } and +1 to those that do not:

            //int[] outputs =
            //{
            //    -1,-1,  // First two sequences are of class -1 (those start with {1,1,1})
            //    1, 1,  // Last two sequences are of class +1  (don't start with {1,1,1})
            //};

            //// At this point, we will have to "flat" out the input sequences from double[][][]
            //// to a double[][] so they can be properly understood by the SVMs. The problem is 
            //// that, normally, SVMs usually expect the data to be comprised of fixed-length 
            //// input vectors and associated class labels. But in this case, we will be feeding
            //// them arbitrary-length sequences of input vectors and class labels associated with
            //// each sequence, instead of each vector.

            //double[][] inputs = new double[sequences.Length][];
            //for (int i = 0; i < sequences.Length; i++)
            //    inputs[i] = Matrix.Concatenate(sequences[i]);


            //// Now we have to setup the Dynamic Time Warping kernel. We will have to
            //// inform the length of the fixed-length observations contained in each
            //// arbitrary-length sequence:
            //// 
            //DynamicTimeWarping kernel = new DynamicTimeWarping(length: 3);

            //// Now we can create the machine. When using variable-length
            //// kernels, we will need to pass zero as the input length:
            //var svm = new KernelSupportVectorMachine(kernel, inputs: 0);


            //// Create the Sequential Minimal Optimization learning algorithm
            //var smo = new SequentialMinimalOptimization(svm, inputs, outputs)
            //{
            //    Complexity = 1.5
            //};

            //// And start learning it!
            //double error = smo.Run(); // error will be 0.0

            // At this point, we should have obtained an useful machine. Let's
            // see if it can understand a few examples it hasn't seem before:

            //double[][] a =
            //{
            //    new double[] { 1, 1, 1 },
            //    new double[] { 7, 2, 5 },
            //    new double[] { 2, 5, 1 },
            //};

            //double[][] b =
            //{
            //    new double[] { 7, 5, 2 },
            //    new double[] { 4, 2, 5 },
            //    new double[] { 1, 1, 1 },
            //};

            //// Following the aforementioned logic, sequence (a) should be
            //// classified as -1, and sequence (b) should be classified as +1.

            //int resultA = System.Math.Sign(svm.Compute(Matrix.Concatenate(a))); // -1
            //int resultB = System.Math.Sign(svm.Compute(Matrix.Concatenate(b))); // +1





            const int featureLength = 8;

            Dictionary<int, Tuple<Dictionary<string, List<List<double[]>>>, Dictionary<string, List<List<double[]>>>>> dict;
            using (System.IO.FileStream fs = System.IO.File.OpenRead(@".\Data.dat"))
            {
                BinaryFormatter bf = new BinaryFormatter();
                dict = bf.Deserialize(fs) as Dictionary<int, Tuple<Dictionary<string, List<List<double[]>>>, Dictionary<string, List<List<double[]>>>>>;
            }

            foreach (KeyValuePair<int, Tuple<Dictionary<string, List<List<double[]>>>, Dictionary<string, List<List<double[]>>>>> kvp in dict)
            {
                List<double[][]> lstSequences = new List<double[][]>();
                List<int> lstOutputs = new List<int>();
                List<string> lstKnownOutputs = new List<string>();

                int nOutputIndex;
                foreach (KeyValuePair<string, List<List<double[]>>> kvpTrain in kvp.Value.Item1)
                {
                    if ((nOutputIndex = lstKnownOutputs.IndexOf(kvpTrain.Key)) < 0)
                    {
                        lstKnownOutputs.Add(kvpTrain.Key);
                        nOutputIndex = lstKnownOutputs.Count - 1;
                    }

                    foreach (List<double[]> gesture in kvpTrain.Value)
                    {
                        lstSequences.Add(gesture.ToArray());
                        lstOutputs.Add(nOutputIndex);
                    }
                }

                double[][][] sequences = lstSequences.ToArray();
                int[] outputs = lstOutputs.ToArray();

                double[][] inputs = new double[sequences.Length][];
                for (int i = 0; i < sequences.Length; i++)
                {
                    inputs[i] = Matrix.Concatenate(sequences[i]);
                }

                DynamicTimeWarping kernel = new DynamicTimeWarping(featureLength);
                KernelSupportVectorMachine svm = new KernelSupportVectorMachine(kernel, inputs: 0);
                var smo = new SequentialMinimalOptimization(svm, inputs, outputs)
                {
                    Complexity = 1.5
                };

                smo.Run();
            }


        }
    }
}
