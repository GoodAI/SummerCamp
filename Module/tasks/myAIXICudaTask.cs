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

namespace AIXIModule
{

    [Description("MC-AIXI-CTW learner on CUDA")]
    class MyAIXICudaTask : MyTask<MyAIXICudaNode>
    {
        public AIXI.BS_RL_Environment env;
        public Dictionary<string, string> options;
        public float explore_rate;
        public float exploration_decay;
        public bool explore;
        public bool explored;

        private MyCudaKernel m_init;
        private MyCudaKernel m_play;


        public override void Init(int nGPU)
        {
            this.options = new Dictionary<string, string>();

            options["ct-depth"] = Owner.ContextTreeDepth.ToString();
            options["agent-horizon"] = Owner.AgentHorizon.ToString();
            options["mc-simulations"] = Owner.MCSimulations.ToString();
            //            options["random-seed"] = Owner.RandomSeed.ToString();

            options["min-action"] = Owner.MinAction.ToString();
            options["max-action"] = Owner.MaxAction.ToString();

            options["min-observation"] = Owner.MinObservation.ToString();
            options["max-observation"] = Owner.MaxObservation.ToString();

            options["min-reward"] = Owner.MinReward.ToString();
            options["max-reward"] = Owner.MaxReward.ToString();

            this.explore_rate = Owner.InitialExploration;
            this.exploration_decay = Owner.ExplorationDecay;
            this.explore = this.explore_rate > 0.0;
            //            MyLog.INFO.WriteLine("### " + this.explore_rate + " " + this.exploration_decay);

            //TODO: proper error handling
            if (this.explore_rate < 0.0 || this.explore_rate > 1.0 || exploration_decay < 0.0 || exploration_decay > 1.0)
            {
                throw new ArgumentException("exploration parameters have to be in [0,1]");
            }

            this.env = new AIXI.BS_RL_Environment(this.options);


            int depth;
            Int32.TryParse(options["ct-depth"], out depth);

            int horizon;
            Int32.TryParse(options["agent-horizon"], out horizon);

            int mc_simulations;
            Int32.TryParse(options["mc-simulations"], out mc_simulations);


            m_init = MyKernelFactory.Instance.Kernel(nGPU, @"AixiKernels", "AixiInitKernel");

            m_init.SetupExecution(1);

            m_init.Run(
                depth,
                mc_simulations,
                horizon,
                Owner.MaxAction-Owner.MinAction,
                Owner.MaxReward-Owner.MinReward,
                Owner.MaxObservation-Owner.MaxObservation
                );
            m_play = MyKernelFactory.Instance.Kernel(nGPU, @"AixiKernels", "AixiPlayKernel");
        
        }

        public override void Execute()
        {
            Owner.Reward.SafeCopyToHost();
            Owner.Input.SafeCopyToHost();

            Owner.Action.Fill(0f);

            Int32 observation = 0;
            for (int j = 0; j < 9; j++)
            {
                if (Owner.Input.Host[j] == 0)
                {
                }
                else if (Owner.Input.Host[j] == 1)
                {
                    observation = observation | (1 << (2 * j));
                }
                else if (Owner.Input.Host[j] == 2)
                {
                    observation = observation | (1 << (2 * j + 1));
                }
            }

            this.env.Observation = observation;
            int rewardUnNormalized = (int)Owner.Reward.Host[0];
            int reward = rewardUnNormalized - this.env.min_reward;
            this.env.Reward = reward;
            MyLog.INFO.WriteLine("observation = " + observation + " and reward =" + rewardUnNormalized);
            m_play.SetupExecution(1);

            m_play.Run(reward, observation, Owner.OutputAction);

            Owner.OutputAction.SafeCopyToHost();

            MyLog.ERROR.WriteLine("output action: " + Owner.OutputAction.Host[0]);

            //agent.ModelUpdatePercept(observation, reward);

            int action;

            //MyLog.INFO.WriteLine("action = " + action);

            /*for (int j = 0; j < 9; j++)
            {
                Owner.Action.Host[j] = 0f;
            }

            Owner.Action.Host[action] = 1f;*/

            //this.env.PerformAction(action);

            /*Owner.Age.Host[0] = this.agent.Age;
            Owner.Observation.Host[0] = observation;
            Owner.Explored.Host[0] = this.explored ? 1 : 0;
            Owner.ExplorationRate.Host[0] = this.explore_rate;
            Owner.TotalReward.Host[0] = (float)agent.TotalReward;

            if (agent.Age > 0)
            {
                Owner.AverageReward.Host[0] = (float)(agent.TotalReward + env.min_reward * agent.Age) / agent.Age;
            }
            else
            {
                Owner.AverageReward.Host[0] = 0.0f;
            }*/

            //Owner.AverageReward.Host[0] = (float)agent.AverageReward();
            //Owner.ModelSize.Host[0] = agent.ModelSize();

            Owner.Action.SafeCopyToDevice();
            Owner.Age.SafeCopyToDevice();
            Owner.Observation.SafeCopyToDevice();
            Owner.Explored.SafeCopyToDevice();
            Owner.ExplorationRate.SafeCopyToDevice();
            Owner.TotalReward.SafeCopyToDevice();
            Owner.AverageReward.SafeCopyToDevice();
            Owner.ModelSize.SafeCopyToDevice();

        }
    }
}
