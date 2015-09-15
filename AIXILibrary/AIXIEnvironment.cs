using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    public abstract class AIXIEnvironment
    {
        public int? action; //TODO maybe?: int? -> int
        public bool is_finished=false;
        public int observation;
        public int reward;
        public Dictionary<string, string> options;
        public int[] valid_actions;
        public int[] valid_observations;
        public int[] valid_rewards;
        
        //method __unicode__

        public AIXIEnvironment(Dictionary<string, string> options) {
            this.options = options;
        }

        public int actionBits() {
            return Utils.BitsNeeded(this.valid_actions.Max());
        }
        public int observationBits()
        {
            return Utils.BitsNeeded(this.valid_observations.Max());
        }
        public int perceptBits()
        {
            return this.observationBits() + this.rewardBits();
        }

        public int rewardBits() {
            return Utils.BitsNeeded(this.valid_rewards.Max());
        }

        public bool isValidAction(int action)
        {
            return this.valid_actions.Contains(action);
        }

        public bool isValidObservation(int observation)
        {
            return this.valid_observations.Contains(observation);
        }
        public bool isValidReward(int reward)
        {
            return this.valid_rewards.Contains(reward);
        }

        public int? maximum_action() {
            //todo: put all maxX together
            //todo: in which sense this should be maximum? - How it is used?
            //  because we are just taking last element, not maximal
            if (this.valid_actions.Length > 0) {
                return this.valid_actions[this.valid_actions.Length-1];
            }
            else
            {
                return null;
            }
        }

        public int? maximum_observation()
        {
            if (this.valid_observations.Length > 0)
            {
                return this.valid_observations[this.valid_observations.Length - 1];
            }
            else
            {
                return null;
            }
        }

        public int? maximum_reward()
        {
            if (this.valid_rewards.Length > 0)
            {
                return this.valid_rewards[this.valid_rewards.Length - 1];
            }
            else
            {
                return null;
            }
        }


        public int? minimum_action() {  //todo: put all minimum_X together
            if (this.valid_actions.Length > 0)
            {
                return this.valid_actions[0]; //TODO? in pyaixi is valid_actions[1] ... Why?
            }
            else {
                return null;
            }
        }
        public int? minimum_observation()
        {
            if (this.valid_observations.Length > 0)
            {
                return this.valid_observations[0];
            }
            else
            {
                return null;
            }
        }
        public int? minimum_reward()
        {
            if (this.valid_rewards.Length > 0)
            {
                return this.valid_rewards[0];
            }
            else
            {
                return null;
            }
        }


        public abstract Tuple<int, int> performAction(int action);

        // TODO:
        //public void print();
    }


}


