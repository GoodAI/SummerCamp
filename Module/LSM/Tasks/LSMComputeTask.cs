using GoodAI.Core;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace LSMModule.LSM.Tasks {
    /// <author>Adr33</author>
    /// <meta>ok</meta>
    /// <status>Work in progress</status>
    /// <summary>Task for inner computing</summary>
    /// <description>
    /// Generates the inner state and internal output of neurons.<br></br>
    /// The main equation for inner state of neurons X in time T is:
    /// innerState[X, T] = (innerState[X, T-1] + A * imageInput + B * 1/N * sum(all edge inputs for X) / (A + B + Threshold),
    /// where N is number of input neurons for neuron X and A/B are constants changeable in BrainSimulator
    /// </description>
    [Description("Compute inner state")]
    class LSMComputeTask : MyTask<LiquidStateMachine> {

        private MyCudaKernel m_LSMResetKernel;
        private MyCudaKernel m_LSMParseInputKernel;
        private MyCudaKernel m_LSMComputeStateKernel;
        private MyCudaKernel m_LSMComputeEdgesKernel;

        public override void Init(int nGPU) {
            m_LSMResetKernel = MyKernelFactory.Instance.Kernel(@"LSMResetKernel");

            m_LSMParseInputKernel = MyKernelFactory.Instance.Kernel(@"LSMParseInputKernel");

            switch (Owner.NeuronType) {
                case LiquidStateMachine.NeuronTypeEnum.IF2:
                    m_LSMComputeStateKernel = MyKernelFactory.Instance.Kernel(@"IF2ComputeStateKernel");
                    break;
                default:
                    m_LSMComputeStateKernel = MyKernelFactory.Instance.Kernel(@"IFComputeStateKernel");
                    break;
            }

            m_LSMComputeEdgesKernel = MyKernelFactory.Instance.Kernel(@"LSMComputeEdgesKernel");
        }

        public override void Execute() {

            float spikeSize = Owner.SpikeSize;
            int spikes;
            if (Owner.Spikes) {
                spikes = 1;
            } else {
                spikes = 0;
            }

            // reset
            m_LSMResetKernel.SetupExecution(Owner.Neurons);
            m_LSMResetKernel.Run(Owner.InitState, Owner.InnerStates, Owner.EdgeInputs, Owner.ImageInput, Owner.Neurons);

            for (int i = 0; i < Owner.InnerCycle; i++) {

                // compute image input
                m_LSMParseInputKernel.SetupExecution(Owner.Input.Count);
                m_LSMParseInputKernel.Run(Owner.Input, Owner.ImageOutput, Owner.ImageInput, spikes, spikeSize, Owner.Input.Count);

                // compute inner states and internal output of neurons
                float thresh = Owner.Threshold;
                m_LSMComputeStateKernel.SetupExecution(Owner.Neurons);
                m_LSMComputeStateKernel.Run(Owner.InitState, Owner.RefractoryState, Owner.Refractory, Owner.EdgeInputs, Owner.ImageInput, Owner.NeuronOutputs, Owner.InnerStates, thresh, Owner.Connectivity, Owner.Neurons);

                // compute value of edges between neurons
                m_LSMComputeEdgesKernel.SetupExecution(Owner.Neurons * Owner.Neurons);
                m_LSMComputeEdgesKernel.Run(Owner.EdgeInputs, Owner.Weights, Owner.NeuronOutputs, spikes, spikeSize, Owner.Neurons, Owner.Neurons * Owner.Neurons);
            }
        }
    }
}
