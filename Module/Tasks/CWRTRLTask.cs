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
            m_outputDeltaKernel.DynamicSharedMemory = (uint)Owner.NeuronGroups;
            m_outputDeltaKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
            m_outputDeltaKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
            m_outputDeltaKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);

            m_inputWeightRTRLDerivativesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\CWRTRLDerivativeKernel", "CWInputWeightsRTRLDerivativesKernel");
            m_inputWeightRTRLDerivativesKernel.SetupExecution(Owner.HIDDEN_UNITS * Owner.HIDDEN_UNITS * Owner.INPUT_UNITS);
            m_inputWeightRTRLDerivativesKernel.DynamicSharedMemory = (uint)Owner.NeuronGroups;
            m_inputWeightRTRLDerivativesKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
            m_inputWeightRTRLDerivativesKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
            m_inputWeightRTRLDerivativesKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
            m_inputWeightRTRLDerivativesKernel.SetConstantVariable("D_NEURONS_PER_GROUP", Owner.NeuronsPerGroup);
            m_inputWeightRTRLDerivativesKernel.SetConstantVariable("D_NEURON_GROUPS", Owner.NeuronGroups);

            m_recurrentWeightRTRLDerivativesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\CWRTRLDerivativeKernel", "CWRecurrentWeightsRTRLDerivativesKernel");
            m_recurrentWeightRTRLDerivativesKernel.SetupExecution(Owner.HIDDEN_UNITS * Owner.HIDDEN_UNITS * Owner.HIDDEN_UNITS);
            m_recurrentWeightRTRLDerivativesKernel.DynamicSharedMemory = (uint)Owner.NeuronGroups;
            m_recurrentWeightRTRLDerivativesKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
            m_recurrentWeightRTRLDerivativesKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
            m_recurrentWeightRTRLDerivativesKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
            m_recurrentWeightRTRLDerivativesKernel.SetConstantVariable("D_NEURONS_PER_GROUP", Owner.NeuronsPerGroup);
            m_recurrentWeightRTRLDerivativesKernel.SetConstantVariable("D_NEURON_GROUPS", Owner.NeuronGroups);

            m_changeInputWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\CWChangeWeightsKernel", "CWChangeInputWeightsKernel");
            m_changeInputWeightsKernel.SetupExecution(Owner.HIDDEN_UNITS * Owner.INPUT_UNITS);
            m_changeInputWeightsKernel.DynamicSharedMemory = (uint)Owner.NeuronGroups;
            m_changeInputWeightsKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
            m_changeInputWeightsKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
            m_changeInputWeightsKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
            m_changeInputWeightsKernel.SetConstantVariable("D_NEURONS_PER_GROUP", Owner.NeuronsPerGroup);
            m_changeInputWeightsKernel.SetConstantVariable("D_NEURON_GROUPS", Owner.NeuronGroups);

            m_changeRecurrentWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\CWChangeWeightsKernel", "CWChangeRecurrentWeightsKernel");
            m_changeRecurrentWeightsKernel.SetupExecution(Owner.HIDDEN_UNITS * Owner.HIDDEN_UNITS);
            m_changeRecurrentWeightsKernel.DynamicSharedMemory = (uint)Owner.NeuronGroups;
            m_changeRecurrentWeightsKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
            m_changeRecurrentWeightsKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
            m_changeRecurrentWeightsKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
            m_changeRecurrentWeightsKernel.SetConstantVariable("D_NEURONS_PER_GROUP", Owner.NeuronsPerGroup);
            m_changeRecurrentWeightsKernel.SetConstantVariable("D_NEURON_GROUPS", Owner.NeuronGroups);

            m_changeOutputWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\CWChangeWeightsKernel", "CWChangeOutputWeightsKernel");
            m_changeOutputWeightsKernel.SetupExecution(Owner.OUTPUT_UNITS * Owner.HIDDEN_UNITS);
            m_changeOutputWeightsKernel.DynamicSharedMemory = (uint)Owner.NeuronGroups;
            m_changeOutputWeightsKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
            m_changeOutputWeightsKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
            m_changeOutputWeightsKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
            m_changeOutputWeightsKernel.SetConstantVariable("D_NEURONS_PER_GROUP", Owner.NeuronsPerGroup);
            m_changeOutputWeightsKernel.SetConstantVariable("D_NEURON_GROUPS", Owner.NeuronGroups);


        }

        public override void Execute()
        {
            // don't copy? use one
            //Owner.InputWeightRTRLDerivatives.CopyToMemoryBlock(Owner.PreviousInputWeightRTRLDerivatives, 0, 0, Owner.InputWeightRTRLDerivatives.Count);
            //Owner.RecurrentWeightRTRLDerivatives.CopyToMemoryBlock(Owner.PreviousRecurrentWeightRTRLDerivatives, 0, 0, Owner.RecurrentWeightRTRLDerivatives.Count);

            m_inputWeightRTRLDerivativesKernel.Run(
                Owner.Input,
                Owner.HiddenActivationDerivatives,
                Owner.RecurrentWeights,
                Owner.InputWeightRTRLDerivatives,
                //Owner.PreviousInputWeightRTRLDerivatives,
                Owner.ActiveGroups,
                Owner.contextByActivations
            );

            m_recurrentWeightRTRLDerivativesKernel.Run(
                Owner.PreviousHiddenActivations,
                Owner.HiddenActivationDerivatives,
                Owner.RecurrentWeights,
                Owner.RecurrentWeightRTRLDerivatives,
                //Owner.PreviousRecurrentWeightRTRLDerivatives,
                Owner.ActiveGroups,
                Owner.contextByActivations
            );

            m_outputDeltaKernel.Run(
                Owner.OutputDeltas,
                Owner.Target,
                Owner.Output,
                Owner.OutputActivationDerivatives
                );

            /*m_outputDeltaKernel.Run(
                Owner.OutputDeltas,
                Owner.Target,
                Owner.OutputActivations,
                Owner.OutputActivationDerivatives
                );*/

            m_changeInputWeightsKernel.Run(
                Owner.InputWeights,
                Owner.InputWeightDeltas,
                Owner.OutputWeights,
                Owner.OutputDeltas,
                Owner.InputWeightRTRLDerivatives,
                LEARNING_RATE,
                MOMENTUM_RATE,
                Owner.ActiveGroups,
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
                Owner.ActiveGroups,
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

            
        }
    }
}