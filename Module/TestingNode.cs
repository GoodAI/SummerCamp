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

        public MyMemoryBlock<int> OutputAction
        {
            get;
            private set;
        }


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
            OutputAction.Count = 1;

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
    
    
}
