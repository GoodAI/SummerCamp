using System;
using System.ComponentModel;
using System.Linq;

using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda;
using YAXLib;
using AIXI;
using System.Collections.Generic;

namespace AIXIModule
{
    /// <author>Jiri Nadvornik</author>
    /// <meta>jn</meta>
    /// <status>Experimental</status>
    /// <summary>
    /// Nothing useful. Just for testing internal functions of AIXI on CUDA.
    /// </summary>
    public class TestingNode : MyWorkingNode
    {
/*        /// <summary>
        /// Base value used by the IncrementTask.
        /// </summary>
        [MyBrowsable, Category("SomeCategory"), YAXSerializableField(DefaultValue = 0.0f)]
        public float IncrementBase { get; set; }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public int InputCount
        {
            get { return (Input != null) ? Input.Count : 0; }
        }*/


        public MyMemoryBlock<int> testInts
        {
            get;
            private set;
        }
        public MyMemoryBlock<float> testFloats
        {
            get;
            private set;
        }


        public MyMemoryBlock<float> LogKt
        {
            get;
            private set;
        }
        public MyMemoryBlock<float> LogProbability
        {
            get;
            private set;
        }
        public MyMemoryBlock<int> NumberOf0S
        {
            get;
            private set;
        }
        public MyMemoryBlock<int> NumberOf1S
        {
            get;
            private set;
        }
        public MyMemoryBlock<int> Child1
        {
            get;
            private set;
        }
        public MyMemoryBlock<int> Child0
        {
            get;
            private set;
        }

        public MyMemoryBlock<int> FreeIndices
        {
            get;
            private set;
        }

        public override void UpdateMemoryBlocks()
        {
            testInts.Count = 100;
            testFloats.Count = 100;

            int lenOfTree = 100;
            LogKt.Count = lenOfTree;
            LogProbability.Count = lenOfTree;
            NumberOf0S.Count = lenOfTree;
            NumberOf1S.Count = lenOfTree;
            Child0.Count = lenOfTree;
            Child1.Count = lenOfTree;
            FreeIndices.Count = lenOfTree;
        }

        /*public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(InputCount != 0, this, "Zero input size is not allowed.");
        }*/

        public MyTestingTask TestingTask { get; private set; }
    }
    
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

            options["ctw-model"] = "ctf";
            options["exploration"] = "0.1";
            options["explore-decay"] = "0.99";
            options["ct-depth"] = "8";
            options["agent-horizon"] = "4";
            options["mc-simulations"] = "200";
            options["terminate-age"] = "500";

            var rightInts =  new int[1000];
            var rightFloats = new float[1000];

            var env=new CoinFlip(options);
            var agent = new MC_AIXI_CTW(env, options);

            var tree = new CTWContextTree(3);
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


            var n0 = tree.Root ;
            var n1 = new CTWContextTreeNode(tree);
            var n2 = new CTWContextTreeNode(tree);
            var n3 = new CTWContextTreeNode(tree);
            var n4 = new CTWContextTreeNode(tree);

            n0.Children[1] = n1;
            n0.Children[0] = n2;
            n2.Children[0] = n4;
            n2.Children[1] = n3;

            var ints = new int[] { 0, 1, 0, 1, 0, 0, 1, 1, 1 };
            tree.update_tree_history(ints);
            tree.update_tree(ints);
            tree.revert_tree(2);

            var intsp = new int[] { 0, 1 };

            double q=tree.Predict(ints);

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



            for (int i=0; i<0; i++){
                if (rightInts[i]!=Owner.testInts.Host[i])
                {
                    MyLog.ERROR.WriteLine("#i "+i+" different: "+rightInts[i] + ";"+Owner.testInts.Host[i]);
                }
                else {
                    MyLog.INFO.WriteLine("#i " + i + " ok: " + rightInts[i] + ";" + Owner.testInts.Host[i]);
                }
                MyLog.Writer.FlushCache();
                if (Math.Abs(rightFloats[i] - Owner.testFloats.Host[i])>0.0001)
                {
                    MyLog.ERROR.WriteLine(">F " + i + " different: " + rightFloats[i] + ";" + Owner.testFloats.Host[i]);
                }
                else
                {
                    MyLog.INFO.WriteLine(">F " + i + " ok: "+ rightFloats[i] + ";" + Owner.testFloats.Host[i]);
                }
                MyLog.Writer.FlushCache();

            }


        }
    }
}
