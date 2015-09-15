using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
//using GoodAI.Core.Utils;

namespace AIXI
{
    public class MonteCarloSearchNode  //beware: This is called Node, not Tree
    {
        public enum nodetype_enum { chance, decision }
        public static int chance_node = (int)nodetype_enum.chance;
        public static int decision_node = (int)nodetype_enum.decision;

        public int type;

        public double exploration_constant = 2.0;
        public double unexplored_bias = 1000000.0;
        public double mean=0.0;

        public int visits = 0;
        public Dictionary<int, MonteCarloSearchNode>  children;
        public MonteCarloSearchNode(int nodetype) {
            this.children = new Dictionary<int, MonteCarloSearchNode>();
            Debug.Assert(nodetype == chance_node || nodetype == decision_node);
            this.type = nodetype;
        }

        public void printBS(int level = 0)
        {
            /*if (level > 2)
            {
                MyLog.DEBUG.WriteLine("...");
                return;
            }
            for (int i = 0; i < level; i++)
            {
                MyLog.DEBUG.Write(".\t");
            }
            MyLog.DEBUG.Write("{0},{1},{2}\t", type, visits, mean);
            foreach (int key in this.children.Keys)
            {
                MyLog.DEBUG.Write("{0}; ", key);
            }
            MyLog.DEBUG.WriteLine("");
            foreach (int key in this.children.Keys)
            {
                this.children[key].printBS(level + 1);
            }*/
        }

        public void print(int level = 0) {
            if (level > 2) {
                Console.WriteLine("...");
                return;
            }
            for (int i = 0; i < level; i++) {
                Console.Write(".\t");            
            }
            Console.Write("{0},{1},{2}\t", type, visits, mean);
            foreach (int key in this.children.Keys) { 
                Console.Write("{0}; ",key);
            }
            Console.WriteLine("");
            foreach (int key in this.children.Keys)
            {
                this.children[key].print(level+1);
            }
         
        }

        public double sample(Agent agent, int horizon, int level=77) {//refact(?): make MCNodeDecision & MCNodeSample &overload this
            double reward = 0.0;

            if (horizon == 0) {
                return (int)reward;
            }
            else if (this.type == chance_node) {
                var percept = agent.GeneratePerceptAndUpdate();
                int observation = percept.Item1;
                int random_reward = percept.Item2;

                if (!this.children.ContainsKey(observation)) {//new node ->add it as decision node
                    this.children[observation] = new MonteCarloSearchNode(decision_node);
                }
                MonteCarloSearchNode observation_child = this.children[observation];

                reward = random_reward + observation_child.sample(agent, horizon-1, level+1);

            }
            else if (this.visits == 0) //unvisited decision node or we have exceeded maximum tree depth
            {
                reward = agent.Playout(horizon);
//                Console.WriteLine("from playout: reward ="+reward);
            }
            else { //Previously visited decision node

                int? action_nullable = this.SelectAction(agent);
                if (action_nullable == null)
                {
                    Debug.Assert(false, "I do not have any action available");
                }
                int action = (int)action_nullable.Value;

                agent.ModelUpdateAction(action);


                if (!this.children.ContainsKey(action)){    //this action is new chance child
                    this.children[action]=new MonteCarloSearchNode(chance_node);
                }
                MonteCarloSearchNode action_child = this.children[action];

                reward = action_child.sample(agent, horizon, level + 1);   //it is not clear if not horizon-1. (asks pyaixi)
            }

            double visitsDouble = (double)this.visits;
            //Console.WriteLine("> {3} - {0}, {1}, {2}", this.mean, reward, (reward + (visitsDouble * this.mean) / (visitsDouble + 1.0)), visitsDouble);
            this.mean = (reward + (visitsDouble*this.mean)) / (1.0 + visitsDouble);
            this.visits = this.visits+1;

            return reward;
        }
        public int? SelectAction(Agent agent){
            Debug.Assert(agent.MaximumReward() != null, "this is weird place, - in selection action");

            double explore_bias = (double)agent.horizon * agent.MaximumReward().Value;//PROC?
            double exploration_numerator = this.exploration_constant * Math.Log(this.visits);
            int? best_action = null;
            double best_priority = double.NegativeInfinity;

            double priority;
            foreach (int action in agent.environment.valid_actions) { 
                MonteCarloSearchNode node=null;
                if (this.children.ContainsKey(action)) {
                    node=this.children[action];
                }
                if (node == null || node.visits == 0) { 
                    // previously unexplored node
                    priority = this.unexplored_bias;    //unexplored bias
                }
                else{
                    priority = node.mean + explore_bias * Math.Sqrt(exploration_numerator / node.visits);
                }

                if (priority > (best_priority+Utils.RandomDouble(0, 0.001))){
                    best_action=action;
                    best_priority=priority;
                }

            }
            return best_action;
        }
    }
}
