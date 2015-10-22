//using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace AIXI
{
    public class MC_AIXI_CTW : Agent
    {
        //Most of high-level logic is here. Implementation of CTW tree is in separate class

        public int Depth;

        public int McSimulations;

        //public CTWContextTreeFast context_tree;
        public IModel ContextTree;


        public MC_AIXI_CTW(AIXIEnvironment env, Dictionary<string, string> options)
            : base(env, options)
        {
            Int32.TryParse(options["ct-depth"], out this.Depth);

            //pick what implementation of CTW tree to use
            if (options.ContainsKey("ctw-model") && options["ctw-model"] == "ctf")
            {
                this.ContextTree = new CTWContextTreeFast(this.Depth);
            }
            else if (!options.ContainsKey("ctw-model") || options["ctw-model"] == "ct")
            {
                this.ContextTree = new CTWContextTree(this.Depth);
            }
            else {
                throw new ArgumentException("unknown ctw-model in options");
            }

            //this.context_tree = new CTWContextTree(this.depth);


            Int32.TryParse(options["agent-horizon"], out this.Horizon);

            Int32.TryParse(options["mc-simulations"], out this.McSimulations);


            this.Reset();
        }

        public override int  ModelSize() {
            return this.ContextTree.get_model_size();
        }

        //encode_X methods are for transforming int into array of its bits.
        // this array is as long as number of bits per action needed.
        //eg: with 4 bits per action (ie: actions are numbered 0-15)
        //          encode_action(13) = [1,0,1,1]
        //   and  with 3 bits per action:
        //          encode_action(13) = [1,0,1]
        public int[] encode_action(int action) {
            return Utils.Encode(action, this.Environment.actionBits());
        }

        public int[] encode_percept(int observation, int reward)
        {
            int[] rewardEncoded = Utils.Encode(reward, this.Environment.rewardBits());
            int[] observationEncoded = Utils.Encode(observation, this.Environment.observationBits());

            var output = new int[observationEncoded.Length + rewardEncoded.Length];
            rewardEncoded.CopyTo(output, 0);
            observationEncoded.CopyTo(output, rewardEncoded.Length);
            
            return output;
        }
        public override void ModelUpdatePercept(int observation, int reward)
        {
            Debug.Assert(this.LastUpdate == ActionUpdate);
            int[] perceptSymbols = this.encode_percept(observation, reward);

            if ((this.LearningPeriod > 0) && (this.Age > this.LearningPeriod))
            {
                this.ContextTree.update_tree_history(perceptSymbols);
            }
            else {
                this.ContextTree.update_tree(perceptSymbols);
                //this.context_tree.update_tree(percept_symbols);

            }
            this.TotalReward += reward;
            this.LastUpdate = PerceptUpdate;
        }

        public int? GenerateAction() {
            throw new NotImplementedException();
//            return this.GenerateRandomAction(); //TODO
        }

        public Tuple<int,int> GeneratePercept() {
            int observation = Utils.RandomElement(this.Environment.ValidObservations);
            int reward = Utils.RandomElement(this.Environment.ValidRewards);
            return new Tuple<int, int>(observation, reward);
        }

        public int decode_reward(int[] symbolList) {
            return Utils.Decode(symbolList, this.Environment.rewardBits());
        }
        public int decode_observation(int[] symbolList)
        {
            return Utils.Decode(symbolList, this.Environment.observationBits());
        }

        public Tuple<int, int> decode_percept(int[] perceptSymbols) {
            int rewardBits = this.Environment.rewardBits();
            int obserservationBits = this.Environment.observationBits();

            int[] rewardSymbols = new int[rewardBits];
            int[] observationSymbols = new int[obserservationBits];

            for (int i = 0; i < rewardBits; i++) {
                rewardSymbols[i] = perceptSymbols[i];
            }
            for (int i = 0; i < obserservationBits; i++)
            {
                observationSymbols[i] = perceptSymbols[rewardBits+i];
            }

            int reward = this.decode_reward(rewardSymbols);

            int observation = this.decode_observation(observationSymbols);
            return new Tuple<int, int>(reward, observation);
        }


        public override Tuple<int, int> GeneratePerceptAndUpdate() {
            int[] perceptSymbols = this.ContextTree.GenerateRandomSymbolsAndUpdate(this.Environment.perceptBits());
            /*for (int k = 0; k < percept_symbols.Length; k++) {
                MyLog.INFO.WriteLine("perc #" + k + " is "+percept_symbols[k]);
                MyLog.Writer.FlushCache();
            }*/
            Tuple<int,int> OandR = this.decode_percept(perceptSymbols);


            int observation = OandR.Item2;
            int reward = OandR.Item1;

            this.TotalReward += reward;
            this.LastUpdate = PerceptUpdate;

            return new Tuple<int, int>(observation, reward);
        }

        public override double Playout(int horizon) {
            double totalReward = 0.0;

            for (int i = 0; i < horizon; i++) {
                int action = this.GenerateRandomAction();
    
                this.ModelUpdateAction(action);
                var or = this.GeneratePerceptAndUpdate();

                totalReward += or.Item2;
                
            }
            return totalReward;
        }

        override public void ModelUpdateAction(int action)
        {
            Debug.Assert(this.Environment.IsValidAction(action));
            Debug.Assert(LastUpdate == PerceptUpdate);

            int[] actionSymbols = this.encode_action(action);

            this.ContextTree.update_tree_history(actionSymbols);

            this.Age += 1;
            this.LastUpdate = ActionUpdate;
        }

        public void model_revert(CtwContextTreeUndo undoInstance)
        {
            while (this.history_size()>undoInstance.HistorySize){
                if (this.LastUpdate == PerceptUpdate)
                {
                    this.ContextTree.revert_tree(this.Environment.perceptBits());

                    this.LastUpdate = ActionUpdate;
                }
                else {
                    this.ContextTree.revert_tree_history(this.Environment.actionBits());
                    //this.context_tree.revert_tree_history(this.environment.actionBits());

                    this.LastUpdate = PerceptUpdate;
                }
            }

            this.Age = undoInstance.Age;
            this.TotalReward = undoInstance.TotalReward;
            this.LastUpdate = undoInstance.LastUpdate;
        }

        override public int Search() {
            CtwContextTreeUndo undoInstance = new CtwContextTreeUndo(this);
            MonteCarloSearchNode searchTree = new MonteCarloSearchNode(MonteCarloSearchNode.DecisionNode);
            for (int i = 0; i < this.McSimulations; i++) {
                searchTree.Sample(this, this.Horizon);
                this.model_revert(undoInstance);
            }

            searchTree.PrintBs();


            int bestAction=-1;
            double bestMean = double.NegativeInfinity;
            foreach (int action in this.Environment.ValidActions) {

                if (!searchTree.Children.ContainsKey(action)) {
                    continue;
                }

                double mean = searchTree.Children[action].Mean + Utils.RandomDouble(0, 0.0001);
                if (mean > bestMean) {
                    bestMean = mean;
                    bestAction = action;
                }
            }
            return bestAction;
        }

        public int history_size()
        {
            return this.ContextTree.History.Count;
        }

        new public void Reset() {
            base.Reset();
            this.ContextTree.Clear();
            base.Reset();

        }

    }
}
