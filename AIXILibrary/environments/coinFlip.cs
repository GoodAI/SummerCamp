using System;
using System.Diagnostics;
using System.Collections.Generic;


namespace AIXI
{
    public class CoinFlip : AIXIEnvironment
    {
        enum ActionsEnum { ATail, AHead };
        enum ObservationsEnum { OTail, OHead };

        enum RewardEnum { RLose=0, RWin=1 };

        //int aTail = (int) actions_enum.aTail;
        //int aHead = (int)actions_enum.aHead;

        public int OTail = (int)ObservationsEnum.OTail;
        public int OHead = (int)ObservationsEnum.OHead;

        public int RLose = (int)RewardEnum.RLose;
        public int RWin = (int)RewardEnum.RWin;

        double _defaultProbability = 0.05;
        double _probability;
        
        Random _rnd = new Random();
        public CoinFlip(Dictionary<string, string> options)
            : base(options)
        {
            ValidActions = (int[])Enum.GetValues(typeof(ActionsEnum));
            ValidObservations = (int[])Enum.GetValues(typeof(ObservationsEnum));
            ValidRewards = (int[])Enum.GetValues(typeof(RewardEnum));
            base.fill_out_bits();

            //todo: OPTIONS -> set probability
            this._probability = _defaultProbability;

            Debug.Assert(this._probability >= 0 && this._probability <= 1, "probability is set outside [0,1]");

            if (this._rnd.NextDouble() < this._probability)
            {
                this.Observation = this.OHead;
            }
            else
            {
                this.Observation = this.OTail;
            }

            this.Reward = 0;
        }

        public override Tuple<int, int> PerformAction(int action)
        {
            Debug.Assert(this.IsValidAction(action), "non-valid action used");

            this.Action = action;
            if (this._rnd.NextDouble() < this._probability)
            {
                this.Observation = this.OHead;
            }
            else
            {
                this.Observation = this.OTail;
            }
            if (action == this.Observation)
            {
                Reward = this.RWin;
            }
            else {
                Reward = this.RLose;
            }

            return new Tuple<int, int>(this.Observation, this.Reward);
        }

    }

}