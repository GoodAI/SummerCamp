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
            var ct = agent.ContextTree;

            Assert.AreEqual(2, ct.History.Count);
            ct.revert_tree_history(1);
            Assert.AreEqual(1, ct.History.Count);
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


            IModel ct = agent.ContextTree;


            ct.update_tree_history(5);
            int[] ints = { 10, 11, 12 };
            ct.update_tree_history(ints);


            Assert.AreEqual(4, ct.History.Count);
            Assert.AreEqual(5, ct.History[0]);
            Assert.AreEqual(10, ct.History[1]);
            Assert.AreEqual(11, ct.History[2]);
            Assert.AreEqual(12, ct.History[3]);

        }
    }
}
