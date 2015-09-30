using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace AIXI
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var options = new Dictionary<string, string>();

            //POSSIBLE OPTIONS:

            //ctw-model: possible values ct/ctf/cuda/random. Which agent
            options["ctw-model"] = "ctf";

            //exploration: (possible values 0.0-1.0), initial probability of exploration.
            //      exploration is probability of taking random action
            options["exploration"] = "0.1";
            
            //explore-decay (possible values 0.0-1.0). Probability of exploration will decay exponentially.
            //      It will be multiplied by explore-decay every cycle
            options["explore-decay"] = "0.99";
            
            //ct-depth (possible values int 1+, common values are 8-100) depth of CTW tree. 
            //      That means, how many bits it will look back while deciding
            options["ct-depth"] = "8";

            //agent-horizon: (values int, 1+) how far into future (in bits) we should try to predict the future.
            //      From predicted future is computed how good this future is.
            options["agent-horizon"] = "4";

            //mc-simulations (possible values 1+). Number of monte carlo simulations to do.
            //      Higher number means better convergence (ie: better data for decision). But also slow speed.
            options["mc-simulations"] = "200";

            //terminate-age: (possible values int 1+)
            //      how many cycles to do, before ending. 0=run forever
            //      cycle = environment gives observation and reward and agent reacts with action
            options["terminate-age"] = "500";
            

            var env = new CoinFlip(options);

            var agent = new MC_AIXI_CTW(env, options);

            var startingTime = DateTime.Now;
            InteractionLoop(agent, env, options);
            var endingTime = DateTime.Now;

            Console.WriteLine("time: {0}", endingTime - startingTime);

            Console.ReadLine();
        }

        //Interaction loop for interaction between agent and environment. 
        //This part is done in BrainSimulator in other version
        // interaction begins with generating observation and reward from environment and giving it to agent
        // agent then generates action and cycle repeats.
        public static void InteractionLoop(Agent agent, AIXIEnvironment env, Dictionary<string, string> options)
        {
            Random rnd;
            if (options.ContainsKey("random-seed"))
            {
                int seed;
                int.TryParse(options["random-seed"], out seed);
                rnd = new Random(seed);
            }
            else
            {
                rnd = new Random();
            }

            // Exploration = try random action
            // probability will decay exponentially as exploreRate * exploreDecay ** round_number
            var exploreRate = 0.0;
            if (options.ContainsKey("exploration"))
            {
                exploreRate = Utils.MyToDouble(options["exploration"]);
            }
            var explore = exploreRate > 0;

            var exploreDecay = 0.0;
            if (options.ContainsKey("explore-decay"))
            {
                exploreDecay = Utils.MyToDouble(options["explore-decay"]);
            }

            Debug.Assert(0.0 <= exploreRate);
            Debug.Assert(0.0 <= exploreDecay && exploreDecay <= 1.0);

            //automatic halting after certain number of rounds
            var terminateAge = 0;
            if (options.ContainsKey("terminate-age"))
            {
                terminateAge = Convert.ToInt32(options["terminate-age"]);
            }
            var terminateCheck = terminateAge > 0;
            Debug.Assert(0 <= terminateAge);

            // when learning period passes, agent will stop changing/improving model and just use it.
            var learningPeriod = 0;
            if (options.ContainsKey("learning-period"))
            {
                learningPeriod = Convert.ToInt32(options["learning-period"]);
            }
            Debug.Assert(0 <= learningPeriod);

            var cycle = 0;
            while (!env.IsFinished)
            {
                
                if (terminateCheck && agent.Age > terminateAge)
                {
                    break;
                }
                var cycleStartTime = DateTime.Now;
                var observation = env.Observation;
                var reward = env.Reward;

                if (learningPeriod > 0 && cycle > learningPeriod)
                {
                    explore = false;
                }

                //give observation and reward to agent.
                agent.ModelUpdatePercept(observation, reward);

                var explored = false;
                int action;

                if (explore && rnd.NextDouble() < exploreRate)
                {
                    explored = true;
                    action = agent.GenerateRandomAction();
                }
                else
                {
                    //get agents response to observation and reward
                    action = agent.Search();
                }

                //pass agent's action to environment
                env.PerformAction(action);
                agent.ModelUpdateAction(action);

                var timeTaken = DateTime.Now - cycleStartTime;

                Console.WriteLine("{0}:\t{1},{2},{3}\t{4},{5}  \t{6},{7}\t>{8},{9}",
                    cycle, observation, reward, action,
                    explored, exploreRate,
                    agent.TotalReward, agent.AverageReward(),
                    timeTaken, agent.ModelSize()
                    );


                if (explore)
                {
                    exploreRate *= exploreDecay;
                }
                cycle += 1;
            }
        }
    }
}