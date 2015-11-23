using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
//using GoodAI.Core.Utils;

using GoodAI.Core.Utils;

// Environment that will be passed to AIXILibrary
// This environment/opponent is Brain Simulator

namespace AIXI
{
    public class BS_RL_Environment : AIXIEnvironment
    {

        public int min_action;
        public int max_action;
        public int actions_num;

        public int min_observation;
        public int max_observation;
        public int observations_num;

        public int min_reward;
        public int max_reward;
        public int rewards_num;


        public BS_RL_Environment(Dictionary<string, string> options)
            : base(options)
        {
            Int32.TryParse(options["min-action"], out this.min_action);
            Int32.TryParse(options["max-action"], out this.max_action);
            this.actions_num = this.max_action - this.min_action + 1;


            Int32.TryParse(options["min-observation"], out this.min_observation);
            Int32.TryParse(options["max-observation"], out this.max_observation);
            this.observations_num = this.max_observation - this.min_observation + 1;

            Int32.TryParse(options["min-reward"], out this.min_reward);
            Int32.TryParse(options["max-reward"], out this.max_reward);
            this.rewards_num = this.max_reward - this.min_reward + 1;

            this.ValidActions = new int[this.actions_num];
            for (int i = 0; i < actions_num; i++) {
                this.ValidActions[i] = i;
            }

            this.ValidObservations= new int[this.observations_num];
            for (int i = 0; i < observations_num; i++)
            {
                this.ValidObservations[i] = i;
            }

            this.ValidRewards = new int[this.rewards_num];
            for (int i = 0; i < rewards_num; i++)
            {
                this.ValidRewards[i] = i;
            }

            base.fill_out_bits();

            this.Reward = 0;
        }

        public override Tuple<int, int> PerformAction(int action)
        {
            this.Action = action;
            //MyLog.INFO.WriteLine("Perf action in TTT: "+action);
            return new Tuple<int, int>(42, 84);
        }
    }
}
