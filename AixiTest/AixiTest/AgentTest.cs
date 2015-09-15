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
        public Dictionary<string, string> options;
        public MC_AIXI_CTW agent;
        public CoinFlip env;
        public AgentTest()
        {

            this.options = new Dictionary<string, string>();

            options["ct-depth"] = "4";
            options["agent-horizon"] = "6";
            options["mc-simulations"] = "200";

            this.env = new CoinFlip(options);

            this.agent = new MC_AIXI_CTW(env, options);
        }


        [TestMethod]
        public void SeveralIterationsTest()
        {//todo there are no asserts here
            int N = 10;
            for (int i = 0; i < N; i++)
            {
                this.agent.ModelUpdatePercept(env.observation, env.reward);
                int? action = agent.search();
                if (action != null)
                {
                    this.env.performAction((int)action);
                    this.agent.ModelUpdateAction((int)action);
                }
            }
            
        }
    }
}
