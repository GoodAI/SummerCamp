using System;
using System.Diagnostics;
using System.Collections.Generic;


namespace AIXI
{
    public class RockPaperScissorsEnvironment : AIXIEnvironment
    {
        enum RPSEnum { Rock, Paper, Scissors };

        enum RewardEnum { RLose=0, RDraw=1, RWin=2 };

        public int Rock = (int)RPSEnum.Rock;
        public int Paper = (int)RPSEnum.Paper;
        public int Scissors = (int)RPSEnum.Scissors;

        public int RLose = (int)RewardEnum.RLose;
        public int RDraw = (int)RewardEnum.RDraw;
        public int RWin = (int)RewardEnum.RWin;

        Random _rnd = new Random();
        public RockPaperScissorsEnvironment(Dictionary<string, string> options)
            : base(options)
        {
            ValidActions = (int[])Enum.GetValues(typeof(RPSEnum));
            ValidObservations = (int[])Enum.GetValues(typeof(RPSEnum));
            ValidRewards = (int[])Enum.GetValues(typeof(RewardEnum));
            base.fill_out_bits();

            this.Observation = Paper;

            this.Reward = 0;
        }

        public override Tuple<int, int> PerformAction(int action)
        {
            //note: here is little confussion because my actions are his observations and vice versa
            Debug.Assert(this.IsValidAction(action), "non-valid action used " + action);

            this.Action = action;

            //Bias in environment: if we won playing rock, we repeat it:
            if ((this.Observation == Rock) && (this.Reward == RLose))
            {
                this.Observation = Rock;
            }
            else
            {
                this.Observation = Utils.RandomElement(this.ValidObservations);
            }

            if (action == this.Observation)
            {
                this.Reward = this.RDraw;
            }
            else if ((action == Rock && Observation == Paper) ||
                (action == Paper && Observation == Scissors) ||
                (action == Scissors && Observation == Rock))
            {//Agent lost; env won
                this.Reward = RLose;
            }
            else
            {//Agent won
                this.Reward = RWin;
            }

            return new Tuple<int, int>(this.Observation, this.Reward);
        }

    }

}