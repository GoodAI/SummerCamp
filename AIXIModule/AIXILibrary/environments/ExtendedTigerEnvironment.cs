using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Runtime.CompilerServices;


namespace AIXI
{
    public class ExtendedTigerEnvironment : AIXIEnvironment
    {
        enum ActionsEnum { AListen, ALeft, ARight, AStand };
        enum ObservationsEnum { ONull, OLeft, ORight };

        enum RewardEnum { RInvalid = 0, RTiger = 0, RStand = 99, RListen = 100, RGold = 130 };

        public int AListen = (int)ActionsEnum.AListen;
        public int ALeft = (int)ActionsEnum.ALeft;
        public int ARight = (int)ActionsEnum.ARight;
        public int AStand = (int)ActionsEnum.AStand;

        public int ONull = (int)ObservationsEnum.ONull;
        public int OLeft = (int)ObservationsEnum.OLeft;
        public int ORight = (int)ObservationsEnum.ORight;

        public int RInvalid = (int)RewardEnum.RInvalid;
        public int RTiger = (int)RewardEnum.RTiger;
        public int RStand = (int)RewardEnum.RStand;
        public int RListen = (int)RewardEnum.RListen;
        public int RGold = (int)RewardEnum.RGold;

        public double default_listen_accuracy = 0.85;
        public double listen_accuracy;

        public int tiger;
        public int gold;
        public bool sitting;

        Random _rnd = new Random();
        public ExtendedTigerEnvironment(Dictionary<string, string> options)
            : base(options)
        {
            ValidActions = (int[])Enum.GetValues(typeof(ActionsEnum));
            ValidObservations = (int[])Enum.GetValues(typeof(ObservationsEnum));
            ValidRewards = (int[])Enum.GetValues(typeof(RewardEnum));
            base.fill_out_bits();


            //todo: make listen_accuracy configurable in options
            this.listen_accuracy = this.default_listen_accuracy;

            Debug.Assert(0.0<= this.listen_accuracy && this.listen_accuracy <= 1.0, "extended tiger listenning accuracy is out of [0-1]");


            this.Observation = this.ONull;
            this.Reward = 0;

            this.reset();
        }

        public void reset()
        {
            // puts tiger and gold to random place, and agent to seat
            // observation/Reward are not changed

            if (Utils.ProbabilisticDecision(0.5))
            {
                this.tiger = ORight;
                this.gold = OLeft;
            }
            else
            {
                this.tiger = OLeft;
                this.gold = ORight;
            }

            this.sitting = true;

        }

        public override Tuple<int, int> PerformAction(int action)
        {
            Debug.Assert(this.IsValidAction(action), "non-valid action used " + action);

            this.Action = action;
            this.Observation = ONull;
            this.Reward = RInvalid;

            if (action == AListen && this.sitting)
            {
                if (Utils.ProbabilisticDecision(this.listen_accuracy))
                {
                    this.Observation = this.tiger;
                }
                else
                {
                    this.Observation = this.gold;
                }
                this.Reward = RListen;
            }
            else if (action == ALeft && !this.sitting)
            {
                if (this.tiger == OLeft)
                {
                    this.Reward = RTiger;

                }
                else
                {
                    this.Reward = RGold;
                }
                this.reset();
            }
            else if (action == ARight && !this.sitting)
            {
                if (this.tiger == ORight)
                {
                    this.Reward = RTiger;

                }
                else
                {
                    this.Reward = RGold;
                }
                this.reset();
            }
            else if (action == AStand && this.sitting)
            {
                this.Reward = this.RStand;
                this.sitting = false;
                //observation stays null
            }




            return new Tuple<int, int>(this.Observation, this.Reward);
        }

    }

}