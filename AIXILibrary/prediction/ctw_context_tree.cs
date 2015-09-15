using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    public class CTWContextTree{
        public int tree_size;
        public List<CTWContextTreeNode> context;
        public int depth;
        public List<int> history;
        public CTWContextTreeNode root;
        public CTWContextTree(int depth)
        {
            context = new List<CTWContextTreeNode>();
            Debug.Assert(depth >= 0);
            this.depth = depth;
            this.history = new List<int> ();
            this.root = new CTWContextTreeNode(this);
            this.tree_size = 1;
        }

        public void clear() { 
            this.history = new List<int>();
            this.root.tree = null; //TODO: is this working - will it free memory? Test!
            //missing?: del self.root
            this.root = new CTWContextTreeNode(this);
            this.tree_size = 1;
            this.context = new List<CTWContextTreeNode>();
        }

        public void GenerateRandomSymbols(int symbol_count) {
            //TODO, I do not understand this one
        }

        public int[] GenerateRandomSymbolsAndUpdate(int symbol_count) {
            int[] symbol_list = new int[symbol_count];
            for (int i = 0; i < symbol_count; i++) {
                int symbol;
                var symbols_to_predict = new int[1];
                symbols_to_predict[0]=1;

                if (Utils.rnd.NextDouble() < this.predict(symbols_to_predict)){
                    symbol=1;
                }
                else{
                    symbol = 0;
                }
                symbol_list[i] = symbol;

                var singletonSymbol = new int[1];
                singletonSymbol[0]=symbol;
                this.update(singletonSymbol);
            }
            return symbol_list;
        }
        public double predict(int[] symbol_list) {
            int symbol_list_length = symbol_list.Length;
            if (this.history.Count + symbol_list_length <= this.depth) {
                return Math.Pow(0.5, symbol_list_length); //note: diff from pyaixi: removing if
            }

            double prob_history = this.root.log_probability;
            this.update(symbol_list);
            double prob_sequence = this.root.log_probability;
            this.revert(symbol_list_length);
            return Math.Exp(prob_sequence - prob_history);
        }

        public void update(int[] symbol_list) {
            foreach (int symbol in symbol_list) {
                if (this.history.Count >= this.depth) {
                    this.update_context();
                    for (int i = this.depth-1; i >=0; i--)
                    {
                        CTWContextTreeNode context_node = this.context[i];
                        context_node.update(symbol);

                    }

                        //foreach (CTWContextTreeNode n in nodes) {
                        //    Console.WriteLine("|| {0}", n.);
                        //}

                        //Console.WriteLine("nodes-length: ", nodes.Count);
                    //foreach(CTWContextTreeNode context_node in nodes){
                    //    context_node.update(symbol);
                    //}
                }
                this.update_history(symbol);
            }

        }


        public void update_context()
        {
            Debug.Assert(this.history.Count >= this.depth, "history is shorter than depth in update_context");
            this.context = new List<CTWContextTreeNode>();
            this.context.Add(this.root);
            CTWContextTreeNode node = this.root;
            int update_depth = 1;
            IEnumerable<int> historyIE = history;
            foreach (int symbol in historyIE.Reverse())
            { //TODO: will this save reversed history to this.history?
                if (node.children.ContainsKey(symbol))
                {
                    node = node.children[symbol];
                }
                else
                {
                    CTWContextTreeNode new_node = new CTWContextTreeNode(this);
                    node.children[symbol] = new_node;
                    this.tree_size += 1;
                    node = new_node;
                }
                this.context.Add(node);
                update_depth += 1;
                if (update_depth > this.depth)
                {
                    break;
                }
            }

        }


        public void revert(int symbol_count = 1)
        {
            for (int i = 0; i < symbol_count; i++) {
                if (this.history.Count == 0) {
                    return;
                }
                int symbol = this.history.Last();
                this.history.RemoveAt(this.history.Count-1);

                if (this.history.Count >= this.depth) {
                    this.update_context();
                    //refact: We do not need to create variable nodes at all
                    for (int j = this.depth - 1; j >= 0; j--) {
                        CTWContextTreeNode node = this.context[j];
                        node.revert(symbol);
                    }
                }
            }
        }

        public void revert_history(int symbol_count) {
            Debug.Assert(symbol_count>=0);
            int history_length = this.history.Count;
            Debug.Assert(history_length >= symbol_count);
            int new_size = history_length - symbol_count;
            this.history = (List<int>)this.history.GetRange(0,new_size);
        }

        public void update_history(int symbol) {
            this.history.Add(symbol);
        }
        public void update_history(int[] symbol_list) {
            foreach (int symbol in symbol_list) {
                this.update_history(symbol);
            }
        }

        public int size() {
            return this.tree_size;
        }

    }
}
