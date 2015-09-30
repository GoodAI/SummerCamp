using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    public abstract class AIXIEnvironment
    {
        public int? Action; //TODO maybe?: int? -> int
        public bool IsFinished=false;
        public int Observation;
        public int Reward;
        public Dictionary<string, string> Options;
        public int[] ValidActions;
        public int[] ValidObservations;
        public int[] ValidRewards;

        public int ActionBits;
        public int ObservationBits;
        public int RewardBits;

        //method __unicode__

        public AIXIEnvironment(Dictionary<string, string> options) {
            this.Options = options;
        }

        public virtual void fill_out_bits(){
            this.ActionBits = Utils.BitsNeeded(this.ValidActions.Max());
            this.ObservationBits = Utils.BitsNeeded(this.ValidObservations.Max());
            this.RewardBits = Utils.BitsNeeded(this.ValidRewards.Max());
        }


        public int actionBits()
        {
            return ActionBits;
        }
        public int observationBits()
        {
            return this.ObservationBits;
        }
        public int perceptBits()
        {
            return this.ObservationBits + this.RewardBits;
        }

        public int rewardBits() {
            return this.RewardBits;
        }

        public bool IsValidAction(int action)
        {
            return this.ValidActions.Contains(action);
        }

        public bool IsValidObservation(int observation)
        {
            return this.ValidObservations.Contains(observation);
        }
        public bool IsValidReward(int reward)
        {
            return this.ValidRewards.Contains(reward);
        }

        public int? maximum_action() {
            //todo: put all maxX together
            //todo: in which sense this should be maximum? - How it is used?
            //  because we are just taking last element, not maximal
            if (this.ValidActions.Length > 0) {
                return this.ValidActions[this.ValidActions.Length-1];
            }
            else
            {
                return null;
            }
        }

        public int? maximum_observation()
        {
            if (this.ValidObservations.Length > 0)
            {
                return this.ValidObservations[this.ValidObservations.Length - 1];
            }
            else
            {
                return null;
            }
        }

        public int? maximum_reward()
        {
            if (this.ValidRewards.Length > 0)
            {
                return this.ValidRewards[this.ValidRewards.Length - 1];
            }
            else
            {
                return null;
            }
        }


        public int? minimum_action() {  //todo: put all minimum_X together
            if (this.ValidActions.Length > 0)
            {
                return this.ValidActions[0]; //TODO? in pyaixi is valid_actions[1] ... Why?
            }
            else {
                return null;
            }
        }
        public int? minimum_observation()
        {
            if (this.ValidObservations.Length > 0)
            {
                return this.ValidObservations[0];
            }
            else
            {
                return null;
            }
        }
        public int? minimum_reward()
        {
            if (this.ValidRewards.Length > 0)
            {
                return this.ValidRewards[0];
            }
            else
            {
                return null;
            }
        }


        public abstract Tuple<int, int> PerformAction(int action);

        // TODO:
        //public void print();
    }


}


