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
    [Description("Create output")]
    class LSMOutputTask : MyTask<LiquidStateMachine> {

        //private MyCudaKernel m_LSMoutputKernel;

        public override void Init(int nGPU) {
            //m_LSMoutputKernel = MyKernelFactory.Instance.Kernel(@"LSMOutputKernel");
        }

        public override void Execute() {
            //m_LSMoutputKernel.SetupExecution(Owner.Neurons);
            //m_LSMoutputKernel.Run(Owner.InnerStates, Owner.NeuronOutputs, Owner.Output, Owner.Threshhold, Owner.Neurons);
            Owner.InnerStates.SafeCopyToHost();
            Owner.NeuronOutputs.SafeCopyToHost();
            int oi = 0;
            for (int i = 0; i < Owner.Neurons; i++) {
                if (Owner.IsInput.Host[i] < 0.5f) {
                    float innerState = Owner.InnerStates.Host[i];
                    float output = Owner.NeuronOutputs.Host[i];
                    Owner.Output.Host[oi++] = Math.Max(innerState, output);
                }
            }

                Owner.FileHeader.Host[0] = Owner.Neurons;
            Owner.FileHeader.Host[1] = Owner.Input.Count;
            Owner.FileHeader.Host[2]=Owner.Connectivity;
            Owner.FileHeader.Host[3]=Owner.Threshhold;
            if(Owner.Spikes){
                Owner.FileHeader.Host[4]=1;
            } else {
                Owner.FileHeader.Host[4]=0;
            }
            Owner.FileHeader.Host[5] = Owner.A;
            Owner.FileHeader.Host[6] = Owner.B;

            Owner.FileHeader.SafeCopyToDevice();
            //Owner.Output.SafeCopyToHost();
            Owner.Output.SafeCopyToDevice();

            //MyLog.DEBUG.WriteLine("--------------------------------------------------------------------------------------");
            //for (int i = 0; i < 28; i++) {
            //    string s = "";
            //    for (int j = 0; j < 28; j++) {
            //        s += outputs[i * 28 + j] + " ";
            //    }
            //    MyLog.DEBUG.WriteLine(s);
            //}
            //MyLog.DEBUG.WriteLine("--------------------------------------------------------------------------------------");
        }
    }
}
