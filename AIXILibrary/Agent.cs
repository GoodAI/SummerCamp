using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    //Common abstract class defining what we need from agent.
    // Two implementations are provided:
    //  RandomAgent: just taking random actions for testing
    //  MC_AIXI_CTW: main implementation. Inside of this class is implemented CTW tree. There are three different implementations (one (ct) is old and slow, one (ctf) is fast and one (cuda) is using CUDA).

    
    abstract public class Agent
    {
        //agents can be in two states: 
        // first they get percept via method ModelUpdatePercept, 
        // then it is possible to get what is best action in this situation, via method Search
        // after is action determined it is needed to say agent that this is action taken via method ModelUpdateAction
        public enum UpdateEnum { ActionUpdate, PerceptUpdate };
        public int ActionUpdate = (int)UpdateEnum.ActionUpdate;
        public int PerceptUpdate = (int)UpdateEnum.PerceptUpdate;
        
        //age (in cycles) of agent
        public int Age = 0;

        //how many bits to try to predict.
        public int Horizon;

        //environment agent is playing against.
        // it is needed just for knowledge about size of data.
        public AIXIEnvironment Environment;

        //main configuration
        public Dictionary<string, string> Options;

        //possible values ActionUpdate/PerceptUpdate
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

        //returns current size of model
        abstract public int ModelSize();

        //tell agent what action it took to update its model
        abstract public void ModelUpdateAction(int action);

        //for telling agent what is response of environment
        abstract public void ModelUpdatePercept(int observation, int reward);

        //try to predict could be response of environment to action of agent
        abstract public Tuple<int, int> GeneratePerceptAndUpdate();

        //Playout: operation from MC tree search
        // it plays game until horizon and then returns what was reward in this play
        abstract public double Playout(int horizon);

        //decide what action is best in current state.
        abstract public int Search();

        //return agent to initial state
        public void Reset() {
            //note: when overriding this method, do not forget to call this base version.
            this.Age = 0;
            this.TotalReward = 0;
            this.LastUpdate = ActionUpdate;
        }
    }
}
