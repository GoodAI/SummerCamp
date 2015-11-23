using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AIXI;
namespace AIXITests
{

    [TestClass]
    public class AgentTest
    {
        public Dictionary<string, string> Options;
        public MC_AIXI_CTW Agent;
        public CoinFlip Env;
        public AgentTest()
        {

            this.Options = new Dictionary<string, string>();

            Options["ct-depth"] = "4";
            Options["agent-horizon"] = "6";
            Options["mc-simulations"] = "200";

            this.Env = new CoinFlip(Options);

            this.Agent = new MC_AIXI_CTW(Env, Options);
        }


        [TestMethod]
        public void SeveralIterationsTest()
        {//todo there are no asserts here
            int n = 10;
            for (int i = 0; i < n; i++)
            {
                this.Agent.ModelUpdatePercept(Env.Observation, Env.Reward);
                int action = Agent.Search();
                
                this.Env.PerformAction((int)action);
                this.Agent.ModelUpdateAction((int)action);
                
            }
            
        }
    }
}
