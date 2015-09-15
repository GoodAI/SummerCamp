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
            Assert.AreEqual(null, e.action);
            Assert.AreEqual(false, e.is_finished);

            if (!(e.observation == e.oHead || e.observation == e.oTail)){
                  Assert.Fail("invalid observation: {0}", e.observation);
            }

            if (e.reward != e.rLose && e.reward != e.rWin) {
                Assert.Fail("invalid reward");
            }

            int [] correct_enum = {0,1};    //correct for actions, observations and rewards
            if (!e.valid_actions.SequenceEqual(correct_enum)) {
                Assert.Fail("valid actions are wrong");
            }
            if (!e.valid_observations.SequenceEqual(correct_enum))
            {
                Assert.Fail("valid actions are wrong");
            }
            if (!e.valid_rewards.SequenceEqual(correct_enum))
            {
                Assert.Fail("valid actions are wrong");
            }

            
            Assert.AreEqual(1, e.actionBits());
            Assert.AreEqual(1, e.observationBits());
            Assert.AreEqual(2, e.perceptBits());
            Assert.AreEqual(1, e.rewardBits());

            Assert.AreEqual(false, e.isValidAction(-1));
            Assert.AreEqual(true, e.isValidAction(1));
            Assert.AreEqual(true, e.isValidAction(0));
            Assert.AreEqual(false, e.isValidAction(2));

            Assert.AreEqual(false, e.isValidObservation(-1));
            Assert.AreEqual(true, e.isValidObservation(1));
            Assert.AreEqual(true, e.isValidObservation(0));
            Assert.AreEqual(false, e.isValidObservation(2));

            Assert.AreEqual(false, e.isValidReward(-1));
            Assert.AreEqual(true, e.isValidReward(1));
            Assert.AreEqual(true, e.isValidReward(0));
            Assert.AreEqual(false, e.isValidReward(2));

            Assert.AreEqual(1, e.maximum_action());
            Assert.AreEqual(1, e.maximum_observation());
            Assert.AreEqual(1, e.maximum_reward());

            Assert.AreEqual(0, e.minimum_action());
            Assert.AreEqual(0, e.minimum_observation());
            Assert.AreEqual(0, e.minimum_reward());

            
            //doing all possible actions
            e.performAction(0);
            Tuple<int, int> res = e.performAction(1);

            Assert.AreEqual(1, e.action);//TODO: test other state vars
            if (e.reward != e.rLose && e.reward != e.rWin)
            {
                Assert.Fail("invalid reward");
            }
            if (e.observation != e.oHead && e.observation != e.oTail)
            {
                Assert.Fail("invalid reward");
            }
            Assert.AreEqual(res.Item1, e.observation);
            Assert.AreEqual(res.Item2, e.reward);


            
        }
    }
}
