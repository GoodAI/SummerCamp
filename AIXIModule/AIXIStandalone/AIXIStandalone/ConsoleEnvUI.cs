using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace AIXI
{
    public class ConsoleEnvUI
    {
        public ConsoleEnvUI()
        {
            var options = new Dictionary<string, string>();

            string layout = 
@"#######
#.....#
#.#.#.#
#.#@#.#
#######";

            var env = new MazeEnvironment(options, layout);

            Console.WriteLine("Max possible action/observation/reward: {0}/{1}/{2}", env.maximum_action(),  env.maximum_observation(), env.maximum_reward());

            Console.WriteLine("Min possible action/observation/action: {0}/{1}/{2}", env.minimum_action(), env.minimum_observation(), env.minimum_reward());


            Console.WriteLine("Bits needed for action/observation/action: {0}/{1}/{2}", env.ActionBits, env.ObservationBits, env.ActionBits);


            int observation;
            int reward;

            while (true)
            {
                Console.Write("> ");
                
                string input_s = Console.ReadLine();
                int input_i = Int32.Parse(input_s);
                if (!env.IsValidAction(input_i))
                {
                    Console.WriteLine("Invalid action, valid are 0-{0}",env.maximum_action());
                    continue;
                }


                var or = env.PerformAction(input_i);
                observation = or.Item1;
                reward = or.Item2;

                env.print();

                Console.WriteLine("observation/reward: {0}/{1}", observation, reward);



            }

        }
        
    }
}