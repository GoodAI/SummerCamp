using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    abstract public class Agent
    {
        public enum UpdateEnum { ActionUpdate, PerceptUpdate };
        public int ActionUpdate = (int)UpdateEnum.ActionUpdate;
        public int PerceptUpdate = (int)UpdateEnum.PerceptUpdate;
        public int Age = 0;
        public int Horizon;  //TODO: this has to be initialised somewhere


        public AIXIEnvironment Environment;
        public Dictionary<string, string> Options;

        public int LastUpdate;
        public int LearningPeriod=0;

        public double TotalReward = 0;

        public Agent(AIXIEnvironment env, Dictionary<string, string> options)
        {
            this.Environment = env;
            this.Options = options;
            this.LastUpdate = ActionUpdate;

            if (options.ContainsKey("learning-period"))
            {
                Int32.TryParse(options["learning-period"], out this.LearningPeriod);
            }
            else {
                this.LearningPeriod = 0;
            }
                

            //todo: this.options

        }

        public double AverageReward() {
            if (this.Age > 0)
            {
                return this.TotalReward / this.Age;
            }
            else {
                return 0.0;
            }
        }

        public int GenerateRandomObservation() {
            return Utils.RandomElement(this.Environment.ValidObservations);
        }
        public int GenerateRandomAction()
        {
            return Utils.RandomElement(this.Environment.ValidActions);
        }
        public int GenerateRandomReward()
        {
            return Utils.RandomElement(this.Environment.ValidRewards);
        }

        public int? MaximumAction() {
            if (this.Environment != null)
            {
                return this.Environment.maximum_action();
            }
            else
            {
                return null;
            }
        }
        public int? MaximumReward()
        {
            if (this.Environment != null)
            {
                return this.Environment.maximum_reward();
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

        abstract public int Search();

        public void Reset() {
            //when overriding this method, do not forget to call this base version.
            this.Age = 0;
            this.TotalReward = 0;
            this.LastUpdate = ActionUpdate;
        }
    }
}
