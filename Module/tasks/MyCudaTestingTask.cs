using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.ComponentModel;
using YAXLib;

using GoodAI.Core.Nodes;
using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using GoodAI.Core.Task;
using System.Threading;
using GoodAI.Core;

using ManagedCuda;
using AIXI;

namespace AIXIModule
{
    /// <summary>
    /// Task that tests various CUDA functions
    /// </summary>
    [Description("Test of various CUDA functions")]
    public class MyTestingTask : MyTask<TestingNode>
    {
        /// <summary>
        /// Testing CUDA versions of functions
        /// </summary>
        [MyBrowsable, Category("SomeCategory"), YAXSerializableField(DefaultValue = 1.0f)]
        public float Increment { get; set; }


        private MyCudaKernel m_play;
        private MyCudaKernel m_init;
        private MyCudaKernel m_test;




        public override void Init(int nGPU)
        {
            //            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"TestingNode", "TestAll");

            
            m_init = MyKernelFactory.Instance.Kernel(nGPU, @"AixiKernels", "AixiInitKernel");

            m_init.SetupExecution(1);

            m_init.Run(
                3,
                200,
                4,
                1,
                1,
                1
                );
            m_play = MyKernelFactory.Instance.Kernel(nGPU, @"AixiKernels", "AixiPlayKernel");
            m_test = MyKernelFactory.Instance.Kernel(nGPU, @"AixiKernels", "AixiTestKernel");

        }

        public override void Execute()
        {
            
            var options = new Dictionary<string, string>();

            //POSSIBLE OPTIONS:

            options["ctw-model"] = "ct";
            options["exploration"] = "0.1";
            options["explore-decay"] = "0.99";
            options["ct-depth"] = "3";
            options["agent-horizon"] = "4";
            options["mc-simulations"] = "200";
            options["terminate-age"] = "500";

            var rightInts = new int[1000];
            var rightFloats = new float[1000];

            var env = new CoinFlip(options);
            var agent = new MC_AIXI_CTW(env, options);

            var tree = agent.ContextTree;
            
            agent.ModelUpdatePercept(0,0);
            agent.ModelUpdateAction(1);
            agent.ModelUpdatePercept(0, 1);
            agent.ModelUpdateAction(0);
            agent.ModelUpdatePercept(1, 1);

            var backup = new CtwContextTreeUndo(agent);
            agent.ModelUpdateAction(0);
            agent.ModelUpdatePercept(1, 0);
            agent.ModelUpdateAction(1);
            agent.ModelUpdatePercept(0, 1);
            agent.model_revert(backup);

            var q = 1 + 1;

            m_test.SetupExecution(1);

            m_test.Run();



            Owner.testInts.SafeCopyToDevice();
            Owner.testFloats.SafeCopyToDevice();
            Owner.LogKt.SafeCopyToDevice();
            Owner.LogProbability.SafeCopyToDevice();
            Owner.NumberOf0S.SafeCopyToDevice();
            Owner.NumberOf1S.SafeCopyToDevice();
            Owner.Child0.SafeCopyToDevice();
            Owner.Child1.SafeCopyToDevice();
            Owner.FreeIndices.SafeCopyToDevice();
            m_play.SetupExecution(1);

            int reward =1;
            int observation=1;

            m_play.Run(reward, observation, Owner.OutputAction);
            m_play.Run(reward, observation, Owner.OutputAction);


            double total_reward = 0;
            double steps = 0;
            while (true) {
                m_play.Run(reward, observation, Owner.OutputAction);
                Owner.OutputAction.SafeCopyToHost();

                int action = Owner.OutputAction.Host[0];
                
                var or = env.PerformAction(action);
                observation = or.Item1;
                reward = or.Item2;

                steps++;
                total_reward += reward;
            }

            Owner.LogKt.SafeCopyToHost();
            Owner.LogProbability.SafeCopyToHost();
            Owner.NumberOf0S.SafeCopyToHost();
            Owner.NumberOf1S.SafeCopyToHost();
            Owner.Child0.SafeCopyToHost();
            Owner.Child1.SafeCopyToHost();
            Owner.testInts.SafeCopyToHost();
            Owner.testFloats.SafeCopyToHost();
            Owner.FreeIndices.SafeCopyToDevice();



            for (int i = 0; i < 0; i++)
            {
                if (rightInts[i] != Owner.testInts.Host[i])
                {
                    MyLog.ERROR.WriteLine("#i " + i + " different: " + rightInts[i] + ";" + Owner.testInts.Host[i]);
                }
                else
                {
                    MyLog.INFO.WriteLine("#i " + i + " ok: " + rightInts[i] + ";" + Owner.testInts.Host[i]);
                }
                MyLog.Writer.FlushCache();
                if (Math.Abs(rightFloats[i] - Owner.testFloats.Host[i]) > 0.0001)
                {
                    MyLog.ERROR.WriteLine(">F " + i + " different: " + rightFloats[i] + ";" + Owner.testFloats.Host[i]);
                }
                else
                {
                    MyLog.INFO.WriteLine(">F " + i + " ok: " + rightFloats[i] + ";" + Owner.testFloats.Host[i]);
                }
                MyLog.Writer.FlushCache();

            }


        }
    }
}
