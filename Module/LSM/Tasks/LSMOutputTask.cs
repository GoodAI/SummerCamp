using GoodAI.Core;
using GoodAI.Core.Task;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;

namespace LSMModule.LSM.Tasks {
    /// <author>Adr33</author>
    /// <meta>ok</meta>
    /// <status>Work in progress</status>
    /// <summary>Task for sending of output</summary>
    /// <description>
    /// Generates the external output of LSM from internal output and inner state of neurons.
    /// </description>
    [Description("Create output")]
    class LSMOutputTask : MyTask<LiquidStateMachine> {

        private MyCudaKernel m_LSMoutputKernel;

        public override void Init(int nGPU) {
            m_LSMoutputKernel = MyKernelFactory.Instance.Kernel(@"LSMOutputKernel");
        }

        public override void Execute() {
            float spikeSize = LiquidStateMachine.SPIKE_SIZE;

            int outputs = Owner.Neurons - Owner.Inputs;

            m_LSMoutputKernel.SetupExecution(outputs);
            m_LSMoutputKernel.Run(Owner.InnerStates, Owner.Output, Owner.OutputsIndex, Owner.Threshold, spikeSize, outputs);

            Owner.Output.SafeCopyToHost();
        }
    }
}
