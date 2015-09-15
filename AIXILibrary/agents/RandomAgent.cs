﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    class RandomAgent: Agent
    {
        public RandomAgent(AIXIEnvironment env, Dictionary<string, string> options)
            : base(env, options)
        {
            this.horizon = 5;  //TODO: what to put here?
        }
        override public int ModelSize() {
            return 0;
        }

        override public void ModelUpdateAction(int action) {
            return;
        }

        override public void ModelUpdatePercept(int observation, int reward) {
            return;
        }

        override public Tuple<int, int> GeneratePerceptAndUpdate() { 
            int observation = Utils.RandomElement(this.environment.valid_observations);
            int reward = Utils.RandomElement(this.environment.valid_rewards);
            return new Tuple<int, int>(observation, reward);
        }

        override public double Playout(int horizon) {
            return horizon * Utils.RandomDouble((int)this.environment.minimum_reward(), (int)this.environment.maximum_reward());//low-todo: remove recasts
        }

        override public int? search() {
            return Utils.RandomElement(this.environment.valid_actions);
        }
    }
}
