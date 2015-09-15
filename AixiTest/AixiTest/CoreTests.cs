using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using AIXI;
namespace AIXITests
{
    [TestClass]
    public class CoreTests
    {
        public CoreTests()
        {




        }
        [TestMethod]
        public void revert_history_Test()
        {
            var options = new Dictionary<string, string>();
            options["ct-depth"] = "4";
            options["agent-horizon"] = "6";
            options["mc-simulations"] = "1";    //original value=300
            options["random-seed"] = "5";
            CoinFlip env = new CoinFlip(options);
            MC_AIXI_CTW agent = new MC_AIXI_CTW(env, options);
            agent.ModelUpdatePercept(1, 1);
            CTWContextTree ct = agent.context_tree;

            Assert.AreEqual(2, ct.history.Count);
            ct.revert_history(1);
            Assert.AreEqual(1, ct.history.Count);
        }

        [TestMethod]
        public void update_history_Test()
        {
            var options = new Dictionary<string, string>();
            options["ct-depth"] = "4";
            options["agent-horizon"] = "6";
            options["mc-simulations"] = "1";    //original value=300
            options["random-seed"] = "5";
            CoinFlip env = new CoinFlip(options);
            MC_AIXI_CTW agent = new MC_AIXI_CTW(env, options);
            
            CTWContextTree ct = agent.context_tree;


            ct.update_history(5);
            int[] ints = { 10, 11, 12 };
            ct.update_history(ints);


            Assert.AreEqual(4, ct.history.Count);
            Assert.AreEqual(5, ct.history[0]);
            Assert.AreEqual(10, ct.history[1]);
            Assert.AreEqual(11, ct.history[2]);
            Assert.AreEqual(12, ct.history[3]);

        }
    }
}
