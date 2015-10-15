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
        public enum NodetypeEnum { Chance, Decision }
        public static int ChanceNode = (int)NodetypeEnum.Chance;
        public static int DecisionNode = (int)NodetypeEnum.Decision;

        public int Type;

        public double ExplorationConstant = 2.0;//LATER: play with this //TODO: put into options
        public double UnexploredBias = 1000000.0;
        public double Mean=0.0;

        public int Visits = 0;
        public Dictionary<int, MonteCarloSearchNode>  Children;
        public MonteCarloSearchNode(int nodetype) {
            this.Children = new Dictionary<int, MonteCarloSearchNode>();
            Debug.Assert(nodetype == ChanceNode || nodetype == DecisionNode);
            this.Type = nodetype;
        }

        public void PrintBs(int level = 0)
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

        public void Print(int level = 0) {
            if (level > 2) {
                Console.WriteLine("...");
                return;
            }
            for (int i = 0; i < level; i++) {
                Console.Write(".\t");            
            }
            Console.Write("{0},{1},{2}\t", Type, Visits, Mean);
            foreach (int key in this.Children.Keys) { 
                Console.Write("{0}; ",key);
            }
            Console.WriteLine("");
            foreach (int key in this.Children.Keys)
            {
                this.Children[key].Print(level+1);
            }
        }

        public double Sample(Agent agent, int horizon, int level=77) {//refact(?): make MCNodeDecision & MCNodeSample &overload this
            //todo: change default value of level, or remove that parameter

            double reward = 0.0;

            if (horizon == 0) {
                return (int)reward;
            }
            else if (this.Type == ChanceNode) {
                var percept = agent.GeneratePerceptAndUpdate();
                int observation = percept.Item1;
                int randomReward = percept.Item2;

                if (!this.Children.ContainsKey(observation)) {//new node ->add it as decision node
                    this.Children[observation] = new MonteCarloSearchNode(DecisionNode);
                }
                MonteCarloSearchNode observationChild = this.Children[observation];

                reward = randomReward + observationChild.Sample(agent, horizon-1, level+1);
            }
            else if (this.Visits == 0) //unvisited decision node or we have exceeded maximum tree depth
            {
                reward = agent.Playout(horizon);
//                Console.WriteLine("from playout: reward ="+reward);
            }
            else { //Previously visited decision node

                int actionNullable = this.SelectAction(agent);
                int action = actionNullable;

                agent.ModelUpdateAction(action);

                if (!this.Children.ContainsKey(action)){    //this action is new chance child
                    this.Children[action]=new MonteCarloSearchNode(ChanceNode);
                }
                MonteCarloSearchNode actionChild = this.Children[action];

                reward = actionChild.Sample(agent, horizon, level + 1);   //it is not clear if not horizon-1. (asks pyaixi)
            }

            double visitsDouble = this.Visits;
            //Console.WriteLine("> {3} - {0}, {1}, {2}", this.mean, reward, (reward + (visitsDouble * this.mean) / (visitsDouble + 1.0)), visitsDouble);
            this.Mean = (reward + (visitsDouble*this.Mean)) / (1.0 + visitsDouble);
            this.Visits = this.Visits+1;

            return reward;
        }
        public int SelectAction(Agent agent){
            Debug.Assert(agent.MaximumReward() != null, "this is weird place, - in selection action");

            double exploreBias = (double)agent.Horizon * agent.MaximumReward().Value;
            double explorationNumerator = this.ExplorationConstant * Math.Log(this.Visits);
            int bestAction = -1;
            double bestPriority = double.NegativeInfinity;

            foreach (int action in agent.Environment.ValidActions) { 
                MonteCarloSearchNode node=null;
                if (this.Children.ContainsKey(action)) {
                    node=this.Children[action];
                }
                double priority;
                if (node == null || node.Visits == 0) { 
                    // previously unexplored node
                    priority = this.UnexploredBias;    //unexplored bias
                }
                else{
                    priority = node.Mean + exploreBias * Math.Sqrt(explorationNumerator / node.Visits);
                }

                if (priority > (bestPriority+Utils.RandomDouble(0, 0.001))){
                    bestAction=action;
                    bestPriority=priority;
                }

            }
            return bestAction;
        }
    }
}
