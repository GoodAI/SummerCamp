﻿using System;
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


namespace AIXIModule
{
    /// <author>Jiri Nadvornik</author>
    /// <meta>jn</meta>
    /// <status>Experimental</status>
    /// <summary>My Tic-Tac-Toe Agent</summary>
    // <description>Uses MC-AIXI-CTW algorithm for reinforcement learning</description>
    class MyAIXINode : MyWorkingNode
    {
        public MyAIXITask Play
        {
            get;
            private set;
        }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input
        {
            get
            {
                return GetInput(0);
            }
        }


        [MyInputBlock(1)]
        public MyMemoryBlock<float> Reward
        {
            get
            {
                return GetInput(1);
            }
        }


        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Action
        {
            get
            {
                return GetOutput(0);
            }

            set
            {
                SetOutput(0, value);
            }
        }
        

        public MyMemoryBlock<int> Age
        {
            get;
            private set;
        }

        public MyMemoryBlock<int> Observation
        {
            get;
            private set;
        }
        
        public MyMemoryBlock<int> Explored
        {
            get;
            private set;
        }

        public MyMemoryBlock<float> ExplorationRate
        {
            get;
            private set;
        }

        public MyMemoryBlock<float> TotalReward
        {
            get;
            private set;
        }

        public MyMemoryBlock<float> AverageReward
        {
            get;
            private set;
        }
        
        public MyMemoryBlock<int> ModelSize
        {
            get;
            private set;
        }

      
        [MyBrowsable, Category("Parameters"), YAXSerializableField(DefaultValue = 200)]
        public int MCSimulations
        {
            get;
            set;
        }

        [MyBrowsable, Category("Parameters"), YAXSerializableField(DefaultValue = 16)]
        public int ContextTreeDepth
        {
            get;
            set;
        }

        [MyBrowsable, Category("Parameters"), YAXSerializableField(DefaultValue = 8)]
        public int AgentHorizon
        {
            get;
            set;
        }
        
        /*[MyBrowsable, Category("Parameters"), YAXSerializableField(DefaultValue = 0)]
        public int RandomSeed
        {
            get;
            set;
        }*/

        [MyBrowsable, Category("Parameters"), YAXSerializableField(DefaultValue = 0)]
        public int MinAction
        {
            get;
            set;
        }
        [MyBrowsable, Category("Parameters"), YAXSerializableField(DefaultValue = 8)]
        public int MaxAction
        {
            get;
            set;
        }
        
        [MyBrowsable, Category("Parameters"), YAXSerializableField(DefaultValue = -10)]
        public int MinReward
        {
            get;
            set;
        }
        [MyBrowsable, Category("Parameters"), YAXSerializableField(DefaultValue = 10)]
        public int MaxReward
        {
            get;
            set;
        }

        [MyBrowsable, Category("Parameters"), YAXSerializableField(DefaultValue = 0)]
        public int MinObservation
        {
            get;
            set;
        }
        [MyBrowsable, Category("Parameters"), YAXSerializableField(DefaultValue = 174762)]
        public int MaxObservation
        {
            get;
            set;
        }

        [MyBrowsable, Category("Parameters"), YAXSerializableField(DefaultValue = 0.9f)]
        public float InitialExploration
        {
            get;
            set;
        }
        [MyBrowsable, Category("Parameters"), YAXSerializableField(DefaultValue = 0.99f)]
        public float ExplorationDecay
        {
            get;
            set;
        }


        public override void Validate(MyValidator validator)
        {
            //MyLog.INFO.WriteLine("In.c = " + Input.Count + " "+ (Input.Count != 9));
            //MyLog.INFO.WriteLine("Out.c = " + Action.Count + " " + (+Action.Count != 9));


            base.Validate(validator);
            validator.AssertError(Input.Count == 9 && Action.Count == 9, this, "Both inputs should have size 9.");
//            validator.AssertWarning(XInput.Count == 1 && YInput.Count == 1, this, "Both inputs should have size 1. Only first value will be considered.");
        }

        public override void UpdateMemoryBlocks()
        {

            Action.Count = 9;
            Age.Count = 1;
            Observation.Count=1;
            Explored.Count = 1;
            ExplorationRate.Count = 1;
            TotalReward.Count = 1;
            AverageReward.Count = 1;
            ModelSize.Count = 1;
        }
    }

    
}
