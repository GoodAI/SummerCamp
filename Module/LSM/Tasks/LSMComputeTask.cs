using GoodAI.Core;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LSMModule.LSM.Tasks {
    /// <author>Adr33</author>
    /// <meta>ok</meta>
    /// <status>Work in progress</status>
    /// <summary>Task for inner computing</summary>
    /// <description>TBA</description>
    [Description("Compute inner state")]
    class LSMComputeTask : MyTask<LiquidStateMachine> {

        private MyCudaKernel m_LSMParseInputKernel;
        private MyCudaKernel m_LSMComputeStateKernel;
        private MyCudaKernel m_LSMComputeEdgesKernel;

        public override void Init(int nGPU) {
            m_LSMParseInputKernel = MyKernelFactory.Instance.Kernel(@"LSMParseInputKernel");
            m_LSMComputeStateKernel = MyKernelFactory.Instance.Kernel(@"LSMComputeStateKernel");
            m_LSMComputeEdgesKernel = MyKernelFactory.Instance.Kernel(@"LSMComputeEdgesKernel");
        }

        public override void Execute() {

            float spikeSize = LiquidStateMachine.SPIKE_SIZE;
            int spikes;
            if (Owner.Spikes) {
                spikes = 1;
            } else {
                spikes = 0;
            }

            for (int i = 0; i < LiquidStateMachine.INNER_CYCLE; i++) {

                // fill with random numbers 0..1
                MyKernelFactory.Instance.GetRandDevice(Owner).GenerateUniform(Owner.ImageSpikeProbabilities.GetDevice(Owner));

                m_LSMParseInputKernel.SetupExecution(Owner.Input.Count);
                m_LSMParseInputKernel.Run(Owner.ImageSpikeProbabilities, Owner.Input, Owner.ImageOutput, Owner.ImageInput, spikes, spikeSize, Owner.Input.Count);

                float thresh = Owner.Threshhold;
                m_LSMComputeStateKernel.SetupExecution(Owner.Neurons);
                m_LSMComputeStateKernel.Run(Owner.A, Owner.B, Owner.EdgeInputs, Owner.ImageInput, Owner.NeuronOutputs, Owner.InnerStates, thresh, Owner.Connectivity, Owner.Neurons);

                m_LSMComputeEdgesKernel.SetupExecution(Owner.Neurons * Owner.Neurons);
                m_LSMComputeEdgesKernel.Run(Owner.EdgeInputs, Owner.Weights, Owner.NeuronOutputs, spikes, spikeSize, Owner.Neurons, Owner.Neurons * Owner.Neurons);
            }
        }
    }
}
