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
        public int depth;
        public int mc_simulations;

        public CTWContextTree context_tree;

        public MC_AIXI_CTW(AIXIEnvironment env, Dictionary<string, string> options)
            : base(env, options)
        {

            Int32.TryParse(options["ct-depth"], out this.depth);

            this.context_tree = new CTWContextTree(this.depth);

            Int32.TryParse(options["agent-horizon"], out this.horizon);

            Int32.TryParse(options["mc-simulations"], out this.mc_simulations);


            this.reset();
        }

        public override int  ModelSize() {
            return this.context_tree.size();
        }

        public int[] encode_action(int action) {
            return Utils.encode(action, this.environment.actionBits());
        }

        public int[] encode_percept(int observation, int reward)
        {
            int[] reward_encoded = Utils.encode(reward, this.environment.rewardBits());
            int[] observation_encoded = Utils.encode(observation, this.environment.observationBits());

            var output = new int[observation_encoded.Length + reward_encoded.Length];
            reward_encoded.CopyTo(output, 0);
            observation_encoded.CopyTo(output, reward_encoded.Length);
            
            return output;
        }
        public override void ModelUpdatePercept(int observation, int reward)
        {
            Debug.Assert(this.last_update == action_update);
            int[] percept_symbols = this.encode_percept(observation, reward);

            if ((this.learning_period > 0) && (this.age > this.learning_period))
            {
                this.context_tree.update_history(percept_symbols);
            }
            else {
                this.context_tree.update(percept_symbols);
            }
            this.total_reward += reward;
            this.last_update = percept_update;

        }

        public int? GenerateAction() {
            return this.GenerateRandomAction(); //TODO
        }

        public Tuple<int,int> GeneratePercept() {
            int observation = Utils.RandomElement(this.environment.valid_observations);
            int reward = Utils.RandomElement(this.environment.valid_rewards);
            return new Tuple<int, int>(observation, reward);
        }

        public int decode_reward(int[] symbol_list) {
            return Utils.decode(symbol_list, this.environment.rewardBits());
        }
        public int decode_observation(int[] symbol_list)
        {
            return Utils.decode(symbol_list, this.environment.observationBits());
        }

        public Tuple<int, int> decode_percept(int[] percept_symbols) {
            int reward_bits = this.environment.rewardBits();
            int obserservation_bits = this.environment.observationBits();

            int[] reward_symbols = new int[reward_bits];
            int[] observation_symbols = new int[obserservation_bits];

            for (int i = 0; i < reward_bits; i++) {
                reward_symbols[i] = percept_symbols[i];
            }
            for (int i = 0; i < obserservation_bits; i++)
            {
                observation_symbols[i] = percept_symbols[reward_bits+i];
            }

            int reward = this.decode_reward(reward_symbols);

            int observation = this.decode_observation(observation_symbols);
            return new Tuple<int, int>(reward, observation);
        }


        public override Tuple<int, int> GeneratePerceptAndUpdate() {
            int[] percept_symbols = this.context_tree.GenerateRandomSymbolsAndUpdate(this.environment.perceptBits());
            /*for (int k = 0; k < percept_symbols.Length; k++) {
                MyLog.INFO.WriteLine("perc #" + k + " is "+percept_symbols[k]);
                MyLog.Writer.FlushCache();
            }*/
            Tuple<int,int> OandR = this.decode_percept(percept_symbols);


            int observation = OandR.Item2;
            int reward = OandR.Item1;

            this.total_reward += reward;
            this.last_update = percept_update;

            return new Tuple<int, int>(observation, reward);
        }

        public override double Playout(int horizon) {
            double total_reward = 0.0;


            for (int i = 0; i < horizon; i++) {

                int action = this.GenerateRandomAction();
    

                this.ModelUpdateAction(action);
                var or = this.GeneratePerceptAndUpdate();

                total_reward += or.Item2;
                
            }
            return total_reward;
        }

        override public void ModelUpdateAction(int action)
        {
            Debug.Assert(this.environment.isValidAction(action));
            Debug.Assert(last_update == percept_update);

            int[] action_symbols = this.encode_action(action);

            this.context_tree.update_history(action_symbols);

            this.age += 1;
            this.last_update = action_update;
        }

        public void model_revert(CTWContextTreeUndo undo_instance)
        {
            while (this.history_size()>undo_instance.history_size){
                if (this.last_update == percept_update)
                {
                    this.context_tree.revert(this.environment.perceptBits());
                    this.last_update = action_update;
                }
                else {
                    this.context_tree.revert_history(this.environment.actionBits());
                    this.last_update = percept_update;
                }
            }

            this.age = undo_instance.age;
            this.total_reward = undo_instance.total_reward;
            this.last_update = undo_instance.last_update;
        }

        override public int? search() {
            CTWContextTreeUndo undo_instance = new CTWContextTreeUndo(this);
            MonteCarloSearchNode search_tree = new MonteCarloSearchNode(MonteCarloSearchNode.decision_node);
            for (int i = 0; i < this.mc_simulations; i++) {
                search_tree.sample(this, this.horizon, 0);
                this.model_revert(undo_instance); //TODO: Uncomment
            }

            search_tree.printBS();


            int best_action=44;//TODO: change to random action
            double best_mean = double.NegativeInfinity;
            foreach (int action in this.environment.valid_actions) {

                if (!search_tree.children.ContainsKey(action)) {
                    continue;
                }

                double mean = search_tree.children[action].mean + Utils.RandomDouble(0, 0.0001);
                if (mean > best_mean) {
                    best_mean = mean;
                    best_action = action;
                }
            }
            return best_action;
        }

        public int history_size() {
            return this.context_tree.history.Count;
        }

        new public void reset() {
            this.context_tree.clear();
            base.reset();

        }

    }
}
