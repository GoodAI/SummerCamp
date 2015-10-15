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
    [Description("Tests various CUDA functions")]
    public class MyTestingTask : MyTask<TestingNode>
    {
        /// <summary>
        /// Testing CUDA versions of functions
        /// </summary>
        [MyBrowsable, Category("SomeCategory"), YAXSerializableField(DefaultValue = 1.0f)]
        public float Increment { get; set; }

        private MyCudaKernel m_kernel;

        public override void Init(int nGPU)
        {
            //            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"TestingNode", "TestAll");
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"TestingNode", "TestAll");

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

            var env = new TigerEnvironment(options);
            var agent = new MC_AIXI_CTW(env, options);

            var tree = agent.ContextTree;
            /*
            var root = tree.Root;
            root.Update(0);
            root.Update(1);
            root.Update(0);
            root.Update(0);
            root.Update(0);

            rightFloats[0] = (float)root.LogKt;
            rightFloats[1] = (float)root.LogProbability;
            rightInts[1] = root.NumberOf0S;
            rightInts[2] = root.NumberOf1S;

            rightFloats[2] = (float)root.LogKtMultiplier(0);

            root.Update(1);
            root.Revert(1);

            rightFloats[3] = (float)root.LogKt;
            rightFloats[4] = (float)root.LogProbability;*/

            Owner.testInts.SafeCopyToDevice();
            Owner.testFloats.SafeCopyToDevice();
            Owner.LogKt.SafeCopyToDevice();
            Owner.LogProbability.SafeCopyToDevice();
            Owner.NumberOf0S.SafeCopyToDevice();
            Owner.NumberOf1S.SafeCopyToDevice();
            Owner.Child0.SafeCopyToDevice();
            Owner.Child1.SafeCopyToDevice();
            Owner.FreeIndices.SafeCopyToDevice();

            /*
            var n0 = tree.Root ;
            var n1 = new CTWContextTreeNode(tree);
            var n2 = new CTWContextTreeNode(tree);
            var n3 = new CTWContextTreeNode(tree);
            var n4 = new CTWContextTreeNode(tree);

            n0.Children[1] = n1;
            n0.Children[0] = n2;
            n2.Children[0] = n4;
            n2.Children[1] = n3;*/

            var ints = new int[] { 1, 1, 0, 1, 0, 0, 1, 1, 1 };
            tree.update_tree_history(ints);
            var ints2 = new int[] { 1, 0, 1, 0 };
            tree.update_tree(ints2);

            tree.revert_tree(2);

            agent.ModelUpdatePercept(2, 11);

            agent.ModelUpdateAction(2);


            m_kernel.SetupExecution(1);

            m_kernel.Run(Owner.testInts, Owner.testFloats, Owner.testInts.Count, Owner.testFloats.Count,
                agent.Depth,
                10, //TODO
                agent.Horizon,
                agent.MaximumAction(),
                agent.MaximumReward(),
                agent.MaximumObservation()
                );

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
