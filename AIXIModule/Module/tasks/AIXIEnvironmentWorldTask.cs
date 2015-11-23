using GoodAI.Core.Task;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AIXI;
using GoodAI.Core.Utils;

namespace AIXIModule
{
    /// <summary>
    /// This task is for passing reaction of agent to environments from AIXILibrary adnd back. 
    /// </summary>
    [Description("Get reaction of AIXIEnvironment")]
    public class AIXIEnvironmentWorldTask : MyTask<AIXIEnvironmentWorld>
    {
        public AIXIEnvironment env;
        
        public override void Init(int nGPU) {
            
            //note: there used to be some environment-dependent options here, instead of these, default values will be used.

            var options = new Dictionary<string, string>();

            Owner.EnvironmentData.SafeCopyToHost();
            switch (Owner.UsedWorld) {
                case AIXIModule.AIXIEnvironmentWorld.myWorlds.CoinFlip:
                    env = new CoinFlip(options);
                    break;
                case AIXIModule.AIXIEnvironmentWorld.myWorlds.Tiger:
                    env = new TigerEnvironment(options);
                    break;
                case AIXIModule.AIXIEnvironmentWorld.myWorlds.ExtendedTiger:
                    env = new ExtendedTigerEnvironment(options);
                    break;
                case AIXIModule.AIXIEnvironmentWorld.myWorlds.RockPaperScissors:
                    env = new RockPaperScissorsEnvironment(options);
                    break;
                case AIXIModule.AIXIEnvironmentWorld.myWorlds.CheeseMaze:
                    env = new CheeseMazeEnvironment(options);
                    break;
                case AIXIModule.AIXIEnvironmentWorld.myWorlds.TicTacToe:
                    env = new MyTttEnvironment(options);
                    break;
                default:
                    MyLog.ERROR.WriteLine("unknown AIXIEnvironment: "+ Owner.UsedWorld);
                    return;
            }
            Owner.EnvironmentData.Host[0]= env.actionBits();
            Owner.EnvironmentData.Host[1]=env.rewardBits();
            Owner.EnvironmentData.Host[2]=env.observationBits();
            Owner.EnvironmentData.Host[3]=env.perceptBits();
            Owner.EnvironmentData.Host[4] = (float) env.minimum_action();//note-todo: using float here can lose precision
            Owner.EnvironmentData.Host[5] = (float) env.minimum_reward();
            Owner.EnvironmentData.Host[6] = (float) env.minimum_observation();
            Owner.EnvironmentData.Host[7] = (float) env.maximum_action();
            Owner.EnvironmentData.Host[8] = (float) env.maximum_reward();
            Owner.EnvironmentData.Host[9] = (float) env.maximum_observation();

            Owner.EnvironmentData.SafeCopyToDevice();
        }


        public override void Execute()
        {
            Owner.Action.SafeCopyToHost();
            Owner.Reward.SafeCopyToHost();
            Owner.Observation.SafeCopyToHost();
            
            int action_i = (int) Owner.Action.Host[0];
            
            var or = this.env.PerformAction(action_i);

            int obervation = or.Item1;
            int reward = or.Item2;


            Owner.Reward.Host[0] = reward;
            Owner.Observation.Host[0] = obervation;

            Owner.Action.SafeCopyToDevice();
            Owner.Reward.SafeCopyToDevice();
            Owner.Observation.SafeCopyToDevice();
        }
    }
}
