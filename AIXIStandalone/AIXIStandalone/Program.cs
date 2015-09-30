using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;


namespace AIXI
{
    class Program
    {
//        public double probability = 0.7;
        static void Main(string[] args)
        {

            var ctf = new CTWContextTreeFast(5);
            var ct = new CTWContextTree(5);

            int[] input = { 1, 1, 0, 1, 1, 0, 0, 0, 1, 1,1,1,0,1 };
            ct.update_tree(input);
            ctf.update_tree(input);
            ct.revert_tree(4);
            ctf.revert_tree(4);
            int[] input2 = {0,0,1};
            ct.update_tree(input2);
            ctf.update_tree(input2);

            if (ctf.compare(ct))
            {
                Console.WriteLine("Stejny!");
            }
            else
            {
                ctf.print_tree();
                Console.WriteLine("----");
                ct.print_tree();
                Console.WriteLine("Ruzny!");
            }



            var options = new Dictionary<string, string>();
            //            environment     = coin_flip
            //# Probability of coin landing heads-up.
            //coin-flip-p     = 0.8
            //exploration     = 0.1
            //explore-decay   = 0.995
            //ct-depth        = 4
            //agent-horizon   = 6
            //learning-period = 5000
            //terminate-age   = 10000
            options["exploration"] = "0.1";
            options["explore-decay"] = "0.99";
            options["ct-depth"] = "6";
            options["agent-horizon"] = "4";
            options["mc-simulations"] = "200";    //original value=300
            options["random-seed"] = "5";
            options["terminate-age"] = "500";
            options["ctw-model"] = "ctf";



//            var rewards = new int[2, 3] { { 1, 2, 6 }, { 3, -4, 5 } }; ;
//            string layout = 
//@"#..
//..#";

            var env = new CoinFlip(options);
            //var env = new TigerEnvironment(options);

            var agent = new MC_AIXI_CTW(env, options);

            //int observation = env.observation;
            //int reward = env.reward;
            //agent.context_tree.root.print();
            //agent.ModelUpdatePercept(observation, reward);
            //int action = env.aListen;
            //var q= env.performAction(action);

            //agent.ModelUpdateAction(action);

            //observation = q.Item1;
            //reward = q.Item2;
            //agent.ModelUpdatePercept(observation, reward);
            //action = env.aRight;
            //q = env.performAction(action);
            //agent.ModelUpdateAction(action);


            //observation = q.Item1;
            //reward = q.Item2;
            //agent.ModelUpdatePercept(observation, reward);
            //action = env.aListen;
            //q = env.performAction(action);
            //agent.ModelUpdateAction(action);

            


            //var undo_instance2 = new CTWContextTreeUndo(agent);
            //var search_tree2 = new MonteCarloSearchNode(MonteCarloSearchNode.decision_node);

            //for (int i = 0; i < 200; i++) {
            //    search_tree2.sample(agent, agent.horizon);
            //    agent.model_revert(undo_instance2);

            //}
            
            //Console.ReadLine();




            //Console.WriteLine("history:");
            //foreach (int symbol in agent.context_tree.history)
            //{
            //    Console.Write("{0};", symbol);
            //}
            //Console.WriteLine();

            //var undo_instance = new CTWContextTreeUndo(agent);
            //var search_tree = new MonteCarloSearchNode(MonteCarloSearchNode.decision_node);

            //DateTime time_beginning = DateTime.Now;
            //for (int i = 0; i < 1000; i++)
            //{
            //    //Console.WriteLine(search_tree.mean);
            //    search_tree.sample(agent, agent.horizon);
            //    agent.model_revert(undo_instance);
            //}
            //DateTime time_end = DateTime.Now;

            //search_tree.print();
            //Console.WriteLine("vvv");

            //agent.context_tree.root.print();


            //Console.WriteLine("history:");
            //foreach (int symbol in agent.context_tree.history)
            //{
            //    Console.Write("{0};", symbol);
            //}
            //Console.WriteLine();

            var startingTime = DateTime.Now;

            InteractionLoop(agent, env, options);
            var endingTime = DateTime.Now;
            Console.WriteLine(endingTime - startingTime);

            //Console.WriteLine(time_end-time_beginning);
            Console.ReadLine();
        }

        static public void InteractionLoop(Agent agent, AIXIEnvironment env, Dictionary<string,string> options) {
            Random rnd;
            if (options.ContainsKey("random-seed"))
            {
                int seed;
                Int32.TryParse(options["random-seed"], out seed);
                rnd = new Random(seed);

            }
            else {
                rnd = new Random();                
            }

            double exploreRate=0.0;
            if (options.ContainsKey("exploration"))
            {
                exploreRate = Utils.MyToDouble(options["exploration"]);
            }
            bool explore = exploreRate > 0;

            double exploreDecay=0.0;
            if (options.ContainsKey("explore-decay"))
            {
                exploreDecay = Utils.MyToDouble(options["explore-decay"]);
            }
  
            Debug.Assert(0.0 <= exploreRate);
            Debug.Assert(0.0 <= exploreDecay && exploreDecay <= 1.0);

            int terminateAge=0;
            if (options.ContainsKey("terminate-age"))
            {
                terminateAge = Convert.ToInt32(options["terminate-age"]);
            }
            bool terminateCheck = terminateAge > 0;
            Debug.Assert(0<=terminateAge);

            int learningPeriod = 0;
            if (options.ContainsKey("learning-period"))
            {
                learningPeriod = Convert.ToInt32(options["learning-period"]);
            }
            Debug.Assert(0 <= learningPeriod);

            int cycle = 0;
            while (!env.IsFinished) {//refact: put computing of total & avg reward here
                if (terminateCheck && agent.Age > terminateAge) {
                    break;
                }
                var cycleStart = DateTime.Now;
                int observation = env.Observation;
                int reward = env.Reward;

                if (learningPeriod > 0 && cycle > learningPeriod) {
                    explore = false;
                }

                agent.ModelUpdatePercept(observation, reward);

                bool explored = false;
                int action;

                if (explore && rnd.NextDouble() < exploreRate)
                {
                    explored = true;
                    action = agent.GenerateRandomAction();
                }
                else {
                    //low-todo: deal with nullable
                    action = (int)agent.Search();
                }

                env.PerformAction(action);
                agent.ModelUpdateAction(action);

                TimeSpan timeTaken = DateTime.Now - cycleStart;

                Console.WriteLine("{0}:\t{1},{2},{3}\t{4},{5}  \t{6},{7}\t>{8},{9}",
                    cycle, observation, reward, action,
                    explored, exploreRate,
                    agent.TotalReward, agent.AverageReward(),
                    timeTaken, agent.ModelSize()
                    );


                if (explore) {
                    exploreRate *= exploreDecay;
                }
                cycle += 1;
            }
        }
    }
}
