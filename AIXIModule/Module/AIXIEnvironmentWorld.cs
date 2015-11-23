using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace AIXIModule
{
    /// <author>Jiri Nadvornik</author>
    /// <meta>jn</meta>
    /// <status>Experimental</status>
    /// <summary>This world makes AIXIEnvironments from AIXILibrary available in Brain Simulator</summary>
    /// <description>Reinforcement learning module using Monte Carlo AIXI CTW node
    ///             
    ///             For more information, see tutorial online.
    /// 
    ///             Inputs:
    ///             <ul>
    ///                 <li>Action - number of action agent pick in last round</li>
    /// 
    ///             </ul>
    ///             Outputs:
    ///             <ul>
    ///                 <li>Observation - number representing bits of last observation</li>
    ///                 <li>Reward - bits representing value of last action ()</li>
    ///                 <li>EnvironmentData - (see docs for description)</li>
    ///             </ul>
    /// </description>
    public class AIXIEnvironmentWorld : MyWorld
    {

        [MyInputBlock(0)]//N: with or without (0)? (I have with, in gridworld it is without)
        public MyMemoryBlock<float> Action //todo: float? how to put int here?
        {
            get
            {
                return GetInput(0);
            }
        }


        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Observation
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

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> Reward
        {
            get
            {
                return GetOutput(1);
            }

            set
            {
                SetOutput(1, value);
            }
        }

        /* This will contain array of these values:
         * 0 - action_bits
         * 1 - reward_bits
         * 2 - observation_bits
         * 3 - perception_bits (sum of two above)
         * 4 - min_action
         * 5 - min_reward
         * 6 - min_observation
         * 7 - max_action
         * 8 - max_reward
         * 9 - max_observation
         * */
        [MyOutputBlock(2)]
        public MyMemoryBlock<float> EnvironmentData
        {
            get
            {
                return GetOutput(2);
            }

            set
            {
                SetOutput(2, value);
            }
        }


        [MyBrowsable, Category("World")]
        [YAXSerializableField(DefaultValue = myWorlds.CoinFlip)]
        public myWorlds UsedWorld
        {
            get
            {
                return mc;
            }
            set
            {
                mc = value;
            }
        }

        public myWorlds mc;
        public enum myWorlds
        {
            CoinFlip = 0,
            Tiger = 1,
            ExtendedTiger = 2,
            RockPaperScissors = 3,
            CheeseMaze = 4,
            TicTacToe = 5
        };


        public AIXIEnvironmentWorldTask reaction
        {
            get;
            private set;
        }
        public override void Validate(MyValidator validator)
        {

            base.Validate(validator);
        }

        public override void UpdateMemoryBlocks()
        {
            Reward.Count = 1;
            Observation.Count = 1;
            EnvironmentData.Count = 10;
        }
    }
}
