using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AIXI;
using System.Linq;
using System.Collections.Generic;

namespace AIXITests
{
    [TestClass]
    public class CoinFlipTest
    {
        [TestMethod]
        public void EnvironmentClassTest()
        {
            var options = new Dictionary<string, string>();
            CoinFlip e = new CoinFlip(options);

            Assert.AreEqual(false, e.IsFinished);

            if (!(e.Observation == e.OHead || e.Observation == e.OTail)){
                  Assert.Fail("invalid observation: {0}", e.Observation);
            }

            if (e.Reward != e.RLose && e.Reward != e.RWin) {
                Assert.Fail("invalid reward");
            }

            int [] correctEnum = {0,1};    //correct for actions, observations and rewards
            if (!e.ValidActions.SequenceEqual(correctEnum)) {
                Assert.Fail("valid actions are wrong");
            }
            if (!e.ValidObservations.SequenceEqual(correctEnum))
            {
                Assert.Fail("valid actions are wrong");
            }
            if (!e.ValidRewards.SequenceEqual(correctEnum))
            {
                Assert.Fail("valid actions are wrong");
            }

            
            Assert.AreEqual(1, e.actionBits());
            Assert.AreEqual(1, e.observationBits());
            Assert.AreEqual(2, e.perceptBits());
            Assert.AreEqual(1, e.rewardBits());

            Assert.AreEqual(false, e.IsValidAction(-1));
            Assert.AreEqual(true, e.IsValidAction(1));
            Assert.AreEqual(true, e.IsValidAction(0));
            Assert.AreEqual(false, e.IsValidAction(2));

            Assert.AreEqual(false, e.IsValidObservation(-1));
            Assert.AreEqual(true, e.IsValidObservation(1));
            Assert.AreEqual(true, e.IsValidObservation(0));
            Assert.AreEqual(false, e.IsValidObservation(2));

            Assert.AreEqual(false, e.IsValidReward(-1));
            Assert.AreEqual(true, e.IsValidReward(1));
            Assert.AreEqual(true, e.IsValidReward(0));
            Assert.AreEqual(false, e.IsValidReward(2));

            Assert.AreEqual(1, e.maximum_action());
            Assert.AreEqual(1, e.maximum_observation());
            Assert.AreEqual(1, e.maximum_reward());

            Assert.AreEqual(0, e.minimum_action());
            Assert.AreEqual(0, e.minimum_observation());
            Assert.AreEqual(0, e.minimum_reward());

            
            //doing all possible actions
            e.PerformAction(0);
            Tuple<int, int> res = e.PerformAction(1);

            Assert.AreEqual(1, e.Action);//TODO: test other state vars
            if (e.Reward != e.RLose && e.Reward != e.RWin)
            {
                Assert.Fail("invalid reward");
            }
            if (e.Observation != e.OHead && e.Observation != e.OTail)
            {
                Assert.Fail("invalid reward");
            }
            Assert.AreEqual(res.Item1, e.Observation);
            Assert.AreEqual(res.Item2, e.Reward);


            
        }
    }
}
