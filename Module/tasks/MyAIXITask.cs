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

    [Description("MC-AIXI-CTW agent")]
    class MyAIXITask : MyTask<MyAIXINode>
    {
        //        public int m_cursor = 0;
        public int i = 0;
        public AIXI.Agent agent;
        public AIXI.BS_RL_Environment env;
        public Dictionary<string, string> options;
        public float explore_rate;
        public float exploration_decay;
        public bool explore;
        public bool explored;
        public int experimental_period;

        public override void Init(int nGPU)
        {
            this.options = new Dictionary<string, string>();

            options["ct-depth"] = Owner.ContextTreeDepth.ToString();
            options["agent-horizon"] = Owner.AgentHorizon.ToString();
            options["mc-simulations"] = Owner.MCSimulations.ToString();
//            options["learning-period"] = Owner.LearningPeriod.ToString();

            options["ctw-model"] = "ctf";

            Int32.TryParse(Owner.LearningPeriod.ToString(), out this.experimental_period);


            if (Owner.UseEnvironmentDataFromHere)
            {   
                //use configuration given by user configuration
                options["min-action"] = Owner.MinAction.ToString();
                options["max-action"] = Owner.MaxAction.ToString();

                options["min-observation"] = Owner.MinObservation.ToString();
                options["max-observation"] = Owner.MaxObservation.ToString();

                options["min-reward"] = Owner.MinReward.ToString();
                options["max-reward"] = Owner.MaxReward.ToString();

            }
            else {
                //use configuration given by environment
                Owner.EnvironmentData.SafeCopyToHost();
                options["min-action"] = Owner.EnvironmentData.Host[4].ToString();
                options["max-action"] = Owner.EnvironmentData.Host[7].ToString();

                options["min-observation"] = Owner.EnvironmentData.Host[6].ToString();
                options["max-observation"] = Owner.EnvironmentData.Host[9].ToString();

                options["min-reward"] = Owner.EnvironmentData.Host[5].ToString();
                options["max-reward"] = Owner.EnvironmentData.Host[8].ToString();
            }
            


            this.explore_rate = Owner.InitialExploration;
            this.exploration_decay = Owner.ExplorationDecay;
            this.explore = this.explore_rate > 0.0;


            this.env = new AIXI.BS_RL_Environment(this.options);

            this.agent = new AIXI.MC_AIXI_CTW(this.env, this.options);
            //            this.m_cursor = 0;
            this.i = 0;

            //            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"SomeNode", "IncrementAll");
        }

        public override void Execute()
        {
            Owner.Reward.SafeCopyToHost();
            Owner.Observation.SafeCopyToHost();

            int observation = (int) Owner.Observation.Host[0];

            this.env.Observation = observation;
            int rewardUnNormalized = (int)Owner.Reward.Host[0];
            int reward = rewardUnNormalized - this.env.min_reward;
            this.env.Reward = reward;

//            MyLog.INFO.WriteLine("observation = " + observation + " and reward =" + rewardUnNormalized);

            agent.ModelUpdatePercept(observation, reward);

            this.explored = false;
            int action;

            bool experiment = false;
            if (this.experimental_period > 0 && this.i < this.experimental_period)
            {
                experiment = true;
            }

            if (experiment || (this.explore && AIXI.Utils.ProbabilisticDecision(this.explore_rate)))
            {
                this.explored = true;
                action = (int)agent.GenerateRandomAction();
            }
            else
            {

                action = (int)agent.Search();
            }

            this.explore_rate *= this.exploration_decay;


            //MyLog.INFO.WriteLine("action = " + action);
            Owner.Action.Host[0] = action;

            this.env.PerformAction(action);
            this.agent.ModelUpdateAction(action);

            Owner.Age.Host[0] = this.agent.Age;
            Owner.ObservationMB.Host[0] = observation;
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
            }

            Owner.AverageReward.Host[0] = (float)agent.AverageReward();
            Owner.ModelSize.Host[0] = agent.ModelSize();

            Owner.Action.SafeCopyToDevice();
            Owner.Age.SafeCopyToDevice();
            Owner.ObservationMB.SafeCopyToDevice();
            Owner.Explored.SafeCopyToDevice();
            Owner.ExplorationRate.SafeCopyToDevice();
            Owner.TotalReward.SafeCopyToDevice();
            Owner.AverageReward.SafeCopyToDevice();
            Owner.ModelSize.SafeCopyToDevice();

            this.i++;
        }
    }

}
