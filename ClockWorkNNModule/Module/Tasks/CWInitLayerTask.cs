using System.Threading.Tasks;
using System.ComponentModel;
using ManagedCuda;
using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using YAXLib;
using System;

namespace CWRNN.Tasks
{
    /// <summary>
    /// Task that increments all data items by a constant calculated as node's IncrementBase + task's Increment
    /// </summary>
    [Description("Initialization of network"), MyTaskInfo(OneShot = true)]
    public class CWInitLayerTask : MyTask<CWRNNLayer>
    {
        /// <summary>
        /// Initialization of weights with random values.
        /// Creating upper triangular matrix for recurrent weights
        /// by setting the lower triangle to zero, that means there
        /// is no connection between the specified units 
        /// in the given direction.
        /// </summary>

        private MyCudaKernel m_kernel;

        public override void Init(int nGPU)
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\SetKernel");
            m_kernel.SetupExecution(1);
        }

        public override void Execute()
        {
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.InputWeights.GetDevice(Owner), 0, 0.25f);
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.RecurrentWeights.GetDevice(Owner), 0, 0.25f);
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.OutputWeights.GetDevice(Owner), 0, 0.25f);

            Owner.RecurrentWeights.SafeCopyToHost();

            setPeriods(Owner.Period);
            Owner.ActiveGroups.Fill(1);

            Owner.Periods.SafeCopyToDevice();

            Owner.RecurrentWeights.SafeCopyToDevice(); 

            Owner.InputWeightDeltas.Fill(0);
            Owner.RecurrentWeightDeltas.Fill(0);
            Owner.OutputWeightDeltas.Fill(0);

            Owner.HiddenActivations.Fill(0);
            Owner.ContextActivations.Fill(0);
            Owner.OutputActivations.Fill(0);
            Owner.PreviousHiddenActivations.Fill(0);

            Owner.HiddenActivationDerivatives.Fill(0);
            Owner.OutputActivationDerivatives.Fill(0);

            Owner.InputWeightRTRLDerivatives.Fill(0);
            Owner.RecurrentWeightRTRLDerivatives.Fill(0);

            Owner.PreviousInputWeightRTRLDerivatives.Fill(0);
            Owner.PreviousRecurrentWeightRTRLDerivatives.Fill(0);


        }


        // Different time periods for activation of unit groups.
        // Set the lower triangle of matrix for recurrent weights to zero.

        public void setPeriods(PeriodEnum periodEnum)
        {
            switch (periodEnum)
            {
                case PeriodEnum.ALL_SAME:
                    for (int i = 0; i < Owner.NeuronGroups; i++)
                    {
                        Owner.Periods.Host[i] = 1;
                        for (int j = 0; j < Owner.NeuronsPerGroup; j++)
                        {
                            for (int k = 0; k < i * Owner.NeuronsPerGroup; k++)
                            {
                                Owner.RecurrentWeights.Host[i * Owner.HIDDEN_UNITS * Owner.NeuronsPerGroup
                                    + j * Owner.HIDDEN_UNITS + k] = 0;
                            }
                        }
                    }
                    return;
                case PeriodEnum.EXPONENTIAL:
                    for (int i = 0; i < Owner.NeuronGroups; i++)
                    {
                        Owner.Periods.Host[i] = (int)Math.Pow(2, i);
                        for (int j = 0; j < Owner.NeuronsPerGroup; j++)
                        {
                            for (int k = 0; k < i * Owner.NeuronsPerGroup; k++)
                            {
                                Owner.RecurrentWeights.Host[i * Owner.HIDDEN_UNITS * Owner.NeuronsPerGroup
                                    + j * Owner.HIDDEN_UNITS + k] = 0;
                            }
                        }
                    }

                    return;
                case PeriodEnum.FIBONACCI:
                    for (int i = 0; i < Owner.NeuronGroups; i++)
                    {
                        if (i == 0 || i == 1)
                        {
                            Owner.Periods.Host[i] = 1;
                        }
                        else
                        {
                            Owner.Periods.Host[i] = Owner.Periods.Host[(i - 1)]
                                + Owner.Periods.Host[(i - 2)];
                        }
                        for (int j = 0; j < Owner.NeuronsPerGroup; j++)
                        {
                            for (int k = 0; k < i * Owner.NeuronsPerGroup; k++)
                            {
                                Owner.RecurrentWeights.Host[i * Owner.HIDDEN_UNITS * Owner.NeuronsPerGroup
                                    + j * Owner.HIDDEN_UNITS + k] = 0;
                            }
                        }
                    }
                    return;
                case PeriodEnum.RANDOM_SERIE:
                    Random rnd = new Random();
                    int last = 1;
                    int number = 1;
                    for (int i = 0; i < Owner.NeuronGroups; i++)
                    {
                        while (number == last)
                        {
                            number = rnd.Next(last, 2 + last);
                        } 
                        if (i == 0)
                        {
                            Owner.Periods.Host[i] = 1;
                        }
                        else
                        {
                            Owner.Periods.Host[i] = number;
                        }
                        for (int j = 0; j < Owner.NeuronsPerGroup; j++)
                        {
                            for (int k = 0; k < i * Owner.NeuronsPerGroup; k++)
                            {
                                Owner.RecurrentWeights.Host[i * Owner.HIDDEN_UNITS * Owner.NeuronsPerGroup
                                    + j * Owner.HIDDEN_UNITS + k] = 0;
                            }
                        }
                        last = number;
                    }
                    return;
                case PeriodEnum.QUADRATIC:
                    for (int i = 0; i < Owner.NeuronGroups; i++)
                    {
                        Owner.Periods.Host[i] = (int)Math.Pow(i + 1, 2);
                        for (int j = 0; j < Owner.NeuronsPerGroup; j++)
                        {
                            for (int k = 0; k < i * Owner.NeuronsPerGroup; k++)
                            {
                                Owner.RecurrentWeights.Host[i * Owner.HIDDEN_UNITS * Owner.NeuronsPerGroup
                                    + j * Owner.HIDDEN_UNITS + k] = 0;
                            }
                        }
                    }
                    return;
            }
        }
    }
}