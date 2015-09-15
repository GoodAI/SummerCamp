using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;


namespace AIXI
{
    class Program
    {
//        public double probability = 0.7;
        static void Main(string[] args)
        {
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
            
            InteractionLoop(agent, env, options);

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

            double explore_rate=0.0;
            if (options.ContainsKey("exploration"))
            {
                explore_rate = Utils.MyToDouble(options["exploration"]);
            }
            bool explore = explore_rate > 0;

            double explore_decay=0.0;
            if (options.ContainsKey("explore-decay"))
            {
                explore_decay = Utils.MyToDouble(options["explore-decay"]);
            }
  
            Debug.Assert(0.0 <= explore_rate);
            Debug.Assert(0.0 <= explore_decay && explore_decay <= 1.0);

            int terminate_age=0;
            if (options.ContainsKey("terminate-age"))
            {
                terminate_age = Convert.ToInt32(options["terminate-age"]);
            }
            bool terminate_check = terminate_age > 0;
            Debug.Assert(0<=terminate_age);

            int learning_period = 0;
            if (options.ContainsKey("learning-period"))
            {
                learning_period = Convert.ToInt32(options["learning-period"]);
            }
            Debug.Assert(0 <= learning_period);

            int cycle = 0;
            while (!env.is_finished) {//refact: put computing of total & avg reward here
                if (terminate_check && agent.age > terminate_age) {
                    break;
                }
                DateTime cycle_start = DateTime.Now;
                int observation = env.observation;
                int reward = env.reward;

                if (learning_period > 0 && cycle > learning_period) {
                    explore = false;
                }

                agent.ModelUpdatePercept(observation, reward);

                bool explored = false;
                int action;

                if (explore && rnd.NextDouble() < explore_rate)
                {
                    explored = true;
                    action = agent.GenerateRandomAction();
                }
                else {
                    action = (int)agent.search(); //low-todo: deal with nullable
                }

                env.performAction(action);
                agent.ModelUpdateAction(action);

                TimeSpan time_taken = DateTime.Now - cycle_start;

                Console.WriteLine("{0}:\t{1},{2},{3}\t{4},{5}  \t{6},{7}\t>{8},{9}",
                    cycle, observation, reward, action,
                    explored, explore_rate,
                    agent.total_reward, agent.AverageReward(),
                    time_taken, agent.ModelSize()
                    );


                if (explore) {
                    explore_rate *= explore_decay;
                }
                cycle += 1;
               // System.Threading.Thread.Sleep(1000);
            }
        }
    }
}
