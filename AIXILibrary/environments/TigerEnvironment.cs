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
        public enum actions_enum { aListen=0, aLeft=1, aRight=2 };
        public enum observations_enum { oNull, oLeft, oRight };
        public enum reward_enum { rEaten = 0, rListen = 99, rGold = 110 };

        public int aListen = (int)actions_enum.aListen;
        public int aLeft = (int)actions_enum.aLeft;
        public int aRight = (int)actions_enum.aRight;

        public int oNull = (int)observations_enum.oNull;
        public int oLeft = (int)observations_enum.oLeft;
        public int oRight = (int)observations_enum.oRight;

        public int rEaten = (int)reward_enum.rEaten;
        public int rListen = (int)reward_enum.rListen;
        public int rGold = (int)reward_enum.rGold;

        public double default_listen_accuracy = 0.85;
        public double listen_accuracy;
        public int tiger;
        public int gold;
        public MyRandom myrnd;
        public TigerEnvironment(Dictionary<string, string> options)
            : base(options)
        {
            valid_actions = Enum.GetValues(typeof(actions_enum)).Cast<int>().ToArray();
            valid_observations = Enum.GetValues(typeof(observations_enum)).Cast<int>().ToArray();
            valid_rewards = Enum.GetValues(typeof(reward_enum)).Cast<int>().ToArray();

            this.myrnd = new MyRandom();


            //low-todo: make listen accuracy configurable in options

            listen_accuracy = this.default_listen_accuracy;
            Debug.Assert(0.0 <= this.listen_accuracy && this.listen_accuracy <= 1);
            this.place_tiger();

            this.observation = this.oNull;
            this.reward = 0;
        }
        public void place_tiger()
        {
//            if (Utils.ProbabilisticDecision(0.5))
            if (this.myrnd.NextDouble() < 0.5)
            {
                this.tiger = oLeft;
                this.gold = oRight;
            }
            else
            {
                this.tiger = oRight;
                this.gold = oLeft;
            }
        }

        public string print() {
            Debug.Assert(this.action!=null, "you must do some action before calling print");
            var action_text = new Dictionary<int, string>();
            action_text.Add(aListen, "listen");
            action_text.Add(aLeft, "left");
            action_text.Add(aRight, "right");

            var observation_text = new Dictionary<int, string>();
            observation_text.Add(oNull, "null");
            observation_text.Add(oLeft, "hear tiger at left door");
            observation_text.Add(oRight, "hear tiger at right door");

            var reward_text = new Dictionary<int, string>();
            reward_text.Add(rEaten, "eaten");
            reward_text.Add(rListen, "listen");
            reward_text.Add(rGold, "gold!");

            string message = string.Format("action = {0}, observation = {1}, reward = {2} ({3})", action_text[(int)this.action],
                observation_text[this.observation],
                reward_text[this.reward],
                this.reward - 100
                );
            return message;
        }

        public override Tuple<int, int> performAction(int action)
        {
            Debug.Assert(this.isValidAction(action));
            this.action = action;

            if (action == aListen)
            {
                this.reward = this.rListen;
                if (this.myrnd.NextDouble() < this.listen_accuracy)
                //if (Utils.ProbabilisticDecision(this.listen_accuracy))
                {
                    this.observation = this.tiger;
                }
                else
                {
                    this.observation = this.gold;
                }
            }
            else {
                if ((action == aLeft && tiger == oLeft) || (action == aRight && tiger == oRight))
                {
                    this.reward = rEaten;
                }
                else {
                    this.reward = rGold;
                }

                this.observation = oNull;
                this.place_tiger();
            }


            return new Tuple<int, int>(this.observation, this.reward);
        }
    }
}
