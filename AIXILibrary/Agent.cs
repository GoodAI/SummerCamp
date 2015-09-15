using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    abstract public class Agent
    {
        public enum update_enum { action_update, percept_update };
        public int action_update = (int)update_enum.action_update;
        public int percept_update = (int)update_enum.percept_update;
        public int age = 0;
        public int horizon;  //TODO: this has to be initialised somewhere


        public AIXIEnvironment environment;
        public Dictionary<string, string> options;

        public int last_update;
        public int learning_period=0;

        public double total_reward = 0;

        public Agent(AIXIEnvironment env, Dictionary<string, string> options)
        {
            this.environment = env;
            this.options = options;
            this.last_update = action_update;

            if (options.ContainsKey("learning-period"))
            {
                Int32.TryParse(options["learning-period"], out this.learning_period);
            }
            else {
                this.learning_period = 0;
            }
                

            //todo: this.options

        }

        public double AverageReward() {
            if (this.age > 0)
            {
                return this.total_reward / this.age;
            }
            else {
                return 0.0;
            }
        }

        public int GenerateRandomObservation() {
            return Utils.RandomElement(this.environment.valid_observations);
        }
        public int GenerateRandomAction()
        {
            return Utils.RandomElement(this.environment.valid_actions);
        }
        public int GenerateRandomReward()
        {
            return Utils.RandomElement(this.environment.valid_rewards);
        }

        public int? MaximumAction() {
            if (this.environment != null)
            {
                return this.environment.maximum_action();
            }
            else
            {
                return null;
            }
        }
        public int? MaximumReward()
        {
            if (this.environment != null)
            {
                return this.environment.maximum_reward();
            }
            else
            {
                return null;
            }
        }

        abstract public int ModelSize();

        abstract public void ModelUpdateAction(int action);

        abstract public void ModelUpdatePercept(int observation, int reward);

        abstract public Tuple<int, int> GeneratePerceptAndUpdate();

        abstract public double Playout(int horizon);

        abstract public int? search();

        public void reset() {
            //when overriding this method, do not forget to call this base version.
            this.age = 0;
            this.total_reward = 0;
            this.last_update = action_update;
        }
    }
}
