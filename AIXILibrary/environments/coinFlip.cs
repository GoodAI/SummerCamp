using System;
using System.Diagnostics;
using System.Collections.Generic;


namespace AIXI
{
    public class CoinFlip : AIXIEnvironment
    {
        enum actions_enum { aTail, aHead };
        enum observations_enum { oTail, oHead };

        enum reward_enum { rLose=0, rWin=1 };

        //int aTail = (int) actions_enum.aTail;
        //int aHead = (int)actions_enum.aHead;

        public int oTail = (int)observations_enum.oTail;
        public int oHead = (int)observations_enum.oHead;

        public int rLose = (int)reward_enum.rLose;
        public int rWin = (int)reward_enum.rWin;

        double default_probability = 0.05;
        double probability;
        
        Random rnd = new Random();
        public CoinFlip(Dictionary<string, string> options)
            : base(options)
        {
            valid_actions = (int[])Enum.GetValues(typeof(actions_enum));
            valid_observations = (int[])Enum.GetValues(typeof(observations_enum));
            valid_rewards = (int[])Enum.GetValues(typeof(reward_enum));

            //todo: OPTIONS -> set probability
            this.probability = default_probability;

            Debug.Assert(this.probability >= 0 && this.probability <= 1, "probability is set outside [0,1]");

            if (this.rnd.NextDouble() < this.probability)
            {
                this.observation = this.oHead;
            }
            else
            {
                this.observation = this.oTail;
            }

            this.reward = 0;
        }

        public override Tuple<int, int> performAction(int action)
        {
            Console.WriteLine("## {0}",action);
            Debug.Assert(this.isValidAction(action), "non-valid action used");

            this.action = action;
            if (this.rnd.NextDouble() < this.probability)
            {
                this.observation = this.oHead;
            }
            else
            {
                this.observation = this.oTail;
            }
            if (action == this.observation)
            {
                reward = this.rWin;
            }
            else {
                reward = this.rLose;
            }

            return new Tuple<int, int>(this.observation, this.reward);
        }

    }

}