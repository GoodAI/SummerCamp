using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    public class TigerEnvironment : AIXIEnvironment
    {
        public enum ActionsEnum { AListen=0, ALeft=1, ARight=2 };
        public enum ObservationsEnum { ONull, OLeft, ORight };
        public enum RewardEnum { REaten = 0, RListen = 99, RGold = 110 };

        public int AListen = (int)ActionsEnum.AListen;
        public int ALeft = (int)ActionsEnum.ALeft;
        public int ARight = (int)ActionsEnum.ARight;

        public int ONull = (int)ObservationsEnum.ONull;
        public int OLeft = (int)ObservationsEnum.OLeft;
        public int ORight = (int)ObservationsEnum.ORight;

        public int REaten = (int)RewardEnum.REaten;
        public int RListen = (int)RewardEnum.RListen;
        public int RGold = (int)RewardEnum.RGold;

        public double DefaultListenAccuracy = 0.85;
        public double ListenAccuracy;
        public int Tiger;
        public int Gold;
        //public MyRandom myrnd;
        public TigerEnvironment(Dictionary<string, string> options)
            : base(options)
        {
            ValidActions = Enum.GetValues(typeof(ActionsEnum)).Cast<int>().ToArray();
            ValidObservations = Enum.GetValues(typeof(ObservationsEnum)).Cast<int>().ToArray();
            ValidRewards = Enum.GetValues(typeof(RewardEnum)).Cast<int>().ToArray();
            base.fill_out_bits();

            //this.myrnd = new MyRandom();


            //low-todo: make listen accuracy configurable via options

            ListenAccuracy = this.DefaultListenAccuracy;
            Debug.Assert(0.0 <= this.ListenAccuracy && this.ListenAccuracy <= 1);
            this.place_tiger();

            this.Observation = this.ONull;
            this.Reward = 0;
        }
        public void place_tiger()
        {
            if (Utils.ProbabilisticDecision(0.5))
            //if (this.myrnd.NextDouble() < 0.5)
            {
                this.Tiger = OLeft;
                this.Gold = ORight;
            }
            else
            {
                this.Tiger = ORight;
                this.Gold = OLeft;
            }
        }

        public string Print() {
            var actionText = new Dictionary<int, string>();
            actionText.Add(AListen, "listen");
            actionText.Add(ALeft, "left");
            actionText.Add(ARight, "right");

            var observationText = new Dictionary<int, string>();
            observationText.Add(ONull, "null");
            observationText.Add(OLeft, "hear tiger at left door");
            observationText.Add(ORight, "hear tiger at right door");

            var rewardText = new Dictionary<int, string>();
            rewardText.Add(REaten, "eaten");
            rewardText.Add(RListen, "listen");
            rewardText.Add(RGold, "gold!");

            string message = string.Format("action = {0}, observation = {1}, reward = {2} ({3})", actionText[this.Action],
                observationText[this.Observation],
                rewardText[this.Reward],
                this.Reward - 100
                );
            return message;
        }

        public override Tuple<int, int> PerformAction(int action)
        {
            Debug.Assert(this.IsValidAction(action));
            this.Action = action;

            if (action == AListen)
            {
                this.Reward = this.RListen;
                //if (this.myrnd.NextDouble() < this.listen_accuracy)
                if (Utils.ProbabilisticDecision(this.ListenAccuracy))
                {
                    this.Observation = this.Tiger;
                }
                else
                {
                    this.Observation = this.Gold;
                }
            }
            else {
                if ((action == ALeft && Tiger == OLeft) || (action == ARight && Tiger == ORight))
                {
                    this.Reward = REaten;
                }
                else {
                    this.Reward = RGold;
                }

                this.Observation = ONull;
                this.place_tiger();
            }


            return new Tuple<int, int>(this.Observation, this.Reward);
        }
    }
}
