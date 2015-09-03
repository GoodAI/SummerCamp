using System.Threading.Tasks;
using System.ComponentModel;

using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda;
using YAXLib;

namespace CWRNN.Tasks
{
    /// <summary>
    /// Task that computes RTRL partial derivatives
    /// </summary>
    [Description("RTRL task"), MyTaskInfo(OneShot = false)]
    public class CWRTRLTask : MyTask<CWRNNLayer>
    {
        /// <summary>
        /// Computing RTRL partial derivatives at each step and updating weights in the network.
        /// </summary>
        /// 

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 0.1f), YAXElementFor("Structure")]
        public float LEARNING_RATE { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
        public float MOMENTUM_RATE { get; set; }

        private MyCudaKernel m_inputWeightRTRLDerivativesKernel;
        private MyCudaKernel m_recurrentWeightRTRLDerivativesKernel;
        private MyCudaKernel m_outputDeltaKernel;
        private MyCudaKernel m_changeInputWeightsKernel;
        private MyCudaKernel m_changeRecurrentWeightsKernel;
        private MyCudaKernel m_changeOutputWeightsKernel;

        public override void Init(int nGPU)
        {
            m_outputDeltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\CWOutputDeltaKernel");
            m_outputDeltaKernel.SetupExecution(Owner.OUTPUT_UNITS);
            m_outputDeltaKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
            m_outputDeltaKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
            m_outputDeltaKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);

            m_inputWeightRTRLDerivativesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\CWRTRLDerivativeKernel", "CWInputWeightsRTRLDerivativesKernel");
            m_inputWeightRTRLDerivativesKernel.SetupExecution(Owner.HIDDEN_UNITS * Owner.HIDDEN_UNITS * Owner.INPUT_UNITS);
            m_inputWeightRTRLDerivativesKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
            m_inputWeightRTRLDerivativesKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
            m_inputWeightRTRLDerivativesKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);

            m_recurrentWeightRTRLDerivativesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\CWRTRLDerivativeKernel", "CWRecurrentWeightsRTRLDerivativesKernel");
            m_recurrentWeightRTRLDerivativesKernel.SetupExecution(Owner.HIDDEN_UNITS * Owner.HIDDEN_UNITS * Owner.HIDDEN_UNITS);
            m_recurrentWeightRTRLDerivativesKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
            m_recurrentWeightRTRLDerivativesKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
            m_recurrentWeightRTRLDerivativesKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);

            m_changeInputWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\CWChangeWeightsKernel", "CWChangeInputWeightsKernel");
            m_changeInputWeightsKernel.SetupExecution(Owner.HIDDEN_UNITS * Owner.INPUT_UNITS);
            m_changeInputWeightsKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
            m_changeInputWeightsKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
            m_changeInputWeightsKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
            //m_changeInputWeightsKernel.SetConstantVariable("neuronsPerGroup", Owner.NeuronsPerGroup);

            m_changeRecurrentWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\CWChangeWeightsKernel", "CWChangeRecurrentWeightsKernel");
            m_changeRecurrentWeightsKernel.SetupExecution(Owner.HIDDEN_UNITS * Owner.HIDDEN_UNITS);
            m_changeRecurrentWeightsKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
            m_changeRecurrentWeightsKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
            m_changeRecurrentWeightsKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
            //m_changeRecurrentWeightsKernel.SetConstantVariable("neuronsPerGroup", Owner.NeuronsPerGroup);

            m_changeOutputWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\CWChangeWeightsKernel", "CWChangeOutputWeightsKernel");
            m_changeOutputWeightsKernel.SetupExecution(Owner.OUTPUT_UNITS * Owner.HIDDEN_UNITS);
            m_changeOutputWeightsKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
            m_changeOutputWeightsKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
            m_changeOutputWeightsKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
            //m_changeOutputWeightsKernel.SetConstantVariable("neuronsPerGroup", Owner.NeuronsPerGroup);


        }

        public override void Execute()
        {
            Owner.InputWeightRTRLDerivatives.CopyToMemoryBlock(Owner.PreviousInputWeightRTRLDerivatives, 0, 0, Owner.InputWeightRTRLDerivatives.Count);
            Owner.RecurrentWeightRTRLDerivatives.CopyToMemoryBlock(Owner.PreviousRecurrentWeightRTRLDerivatives, 0, 0, Owner.RecurrentWeightRTRLDerivatives.Count);

            m_inputWeightRTRLDerivativesKernel.Run(
                Owner.Input,
                Owner.HiddenActivationDerivatives,
                Owner.RecurrentWeights,
                Owner.InputWeightRTRLDerivatives,
                Owner.PreviousInputWeightRTRLDerivatives,
                Owner.Periods,
                Owner.MySimulationSteps,
                Owner.contextByActivations
            );

            m_recurrentWeightRTRLDerivativesKernel.Run(
                Owner.PreviousHiddenActivations,
                Owner.HiddenActivationDerivatives,
                Owner.RecurrentWeights,
                Owner.RecurrentWeightRTRLDerivatives,
                Owner.PreviousRecurrentWeightRTRLDerivatives,
                Owner.Periods,
                Owner.MySimulationSteps,
                Owner.contextByActivations
            );

            m_outputDeltaKernel.Run(
                Owner.OutputDeltas,
                Owner.Target,
                Owner.OutputActivations,
                Owner.OutputActivationDerivatives
                );

            m_changeInputWeightsKernel.Run(
                Owner.InputWeights,
                Owner.InputWeightDeltas,
                Owner.OutputWeights,
                Owner.OutputDeltas,
                Owner.InputWeightRTRLDerivatives,
                LEARNING_RATE,
                MOMENTUM_RATE,
                Owner.Periods,
                Owner.MySimulationSteps,
                Owner.contextByActivations
                );

            m_changeRecurrentWeightsKernel.Run(
                Owner.RecurrentWeights,
                Owner.RecurrentWeightDeltas,
                Owner.OutputWeights,
                Owner.OutputDeltas,
                Owner.RecurrentWeightRTRLDerivatives,
                LEARNING_RATE,
                MOMENTUM_RATE,
                Owner.Periods,
                Owner.MySimulationSteps,
                Owner.contextByActivations
                );

            m_changeOutputWeightsKernel.Run(
                Owner.OutputWeights,
                Owner.OutputWeightDeltas,
                Owner.OutputDeltas,
                Owner.HiddenActivations,
                LEARNING_RATE,
                MOMENTUM_RATE
                );

            // Setting lower triangle back to zero.

            if (Owner.contextByActivations == 0)
            {
                Owner.RecurrentWeights.SafeCopyToHost();
                for (int i = 0; i < Owner.NeuronGroups; i++)
                {
                    for (int j = 0; j < Owner.NeuronsPerGroup; j++)
                    {
                        for (int k = 0; k < i * Owner.NeuronsPerGroup; k++)
                        {
                            Owner.RecurrentWeights.Host[i * Owner.HIDDEN_UNITS * Owner.NeuronsPerGroup
                                + j * Owner.HIDDEN_UNITS + k] = 0;
                        }
                    }
                }
                Owner.RecurrentWeights.SafeCopyToDevice();
            }
            Owner.MySimulationSteps++;
        }
    }
}