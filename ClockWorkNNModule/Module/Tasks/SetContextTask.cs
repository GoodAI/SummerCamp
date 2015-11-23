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
    /// Task that prepares context for the not newly activated neurons.
    /// </summary>
    [Description("Activation context task"), MyTaskInfo(OneShot = false)]
    public class SetContextTask : MyTask<CWRNNLayer>
    {

        private MyCudaKernel m_setContextKernel;

        public override void Init(int nGPU)
        {
            m_setContextKernel = MyKernelFactory.Instance.Kernel(nGPU, @"\CWSetContext");
            m_setContextKernel.SetupExecution(Owner.HIDDEN_UNITS * Owner.NeuronGroups);
            m_setContextKernel.SetConstantVariable("D_NEURONS_PER_GROUP", Owner.NeuronsPerGroup);
            m_setContextKernel.SetConstantVariable("D_NEURON_GROUPS", Owner.NeuronGroups);
            m_setContextKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
        }

        public override void Execute()
        {
            m_setContextKernel.Run(Owner.HiddenActivations, Owner.ContextActivations, Owner.ActiveGroups);

        }
    }
}
