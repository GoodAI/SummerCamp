using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    public class CTWContextTreeNode
    {
        public double log_half = Math.Log(0.5);
        public Dictionary<int, CTWContextTreeNode> children;
        public CTWContextTree tree;
        public double log_kt=0.0;
        public double log_probability = 0.0;//log of prob==0 -> prob ==1. Is this true?
        public int numberOf0s=0;
        public int numberOf1s=0;
        public CTWContextTreeNode(CTWContextTree tree){
            this.children = new Dictionary<int, CTWContextTreeNode>();
            this.tree = tree;

        }

        public void print(int level = 0) {
            for (int i = 0; i < level; i++) {
                Console.Write("    ");
            
            }
            Console.Write("{0}, {1}, {2}-{3}   ", this.log_kt, this.log_probability, this.SymbolCount(0), this.SymbolCount(1));
            foreach (KeyValuePair<int, CTWContextTreeNode> item in this.children) {
                Console.Write("{0}: {1},  ", item.Key, item.Value);
            }
            Console.WriteLine();
            foreach (KeyValuePair<int, CTWContextTreeNode> item in this.children)
            {
                item.Value.print(level + 1);
            }
        
        }

        public bool IsLeaf() {
            return this.children.Count() == 0;
        }
        public int SymbolCount(int symbol) { 
            if (symbol == 0){
                return this.numberOf0s;
            }
            else if (symbol == 1) {
                return this.numberOf1s;
            }
            Debug.Assert(false, "Only symbols 0 and 1 are allowed");
            return -1;
        }

        public double LogKTMultiplier(int symbol) {
            double numerator = this.SymbolCount(symbol)+0.5;
            double denominator = this.visits()+1;
            return Math.Log(numerator / denominator);
        }

        public int visits() {
            return this.numberOf0s + this.numberOf1s;
        }

        public void setSymbolCount(int symbol, int newValue){
            if (symbol==0){
                this.numberOf0s=newValue;
            }
            else if (symbol == 1)
            {
                this.numberOf1s = newValue;
            }
            else
            {
                Console.WriteLine("bad value: {0}", symbol);
                Debug.Assert(false, "trying to set symbol other than 0/1");
            }
        }

        public void revert(int symbol) {
            int this_symbol_count = this.SymbolCount(symbol);
            if (this_symbol_count > 1)
            {
                this.setSymbolCount(symbol, this_symbol_count-1);

            }
            else {
                this.setSymbolCount(symbol, 0);
            }

            if (this.children.ContainsKey(symbol) && children[symbol].visits()==0) {    //note: this piece was not tested with rest of class
                this.tree.tree_size -= children[symbol].size();
                this.children.Remove(symbol);    //low-todo: check if this truly free memory

            }

            this.log_kt -= this.LogKTMultiplier(symbol);
            this.UpdateLogProbability();
        }

        public void update(int symbol) {

            this.log_kt += this.LogKTMultiplier(symbol);
            this.UpdateLogProbability();
            if (symbol == 0) {
                this.numberOf0s += 1;
            }
            else if(symbol==1){
                this.numberOf1s += 1;
            }
        }

        public void UpdateLogProbability() {
            if (this.IsLeaf())
            {
                this.log_probability = this.log_kt;
            }
            else {
                double log_child_probability = 0;
                foreach (CTWContextTreeNode child in this.children.Values) {
                    //beware: this is not best way of doing summation of doubles. We will see if this will matter...
                    // (eg: python has math.fsum)
                    log_child_probability += child.log_probability;
                }

                //for better numerical results
                double a = Math.Max(this.log_kt, log_child_probability);
                double b = Math.Min(this.log_kt, log_child_probability);

                this.log_probability = this.log_half + a + Utils.log1p(Math.Exp(b - a));
            }
        }

        public int size() {
            int count = 1;
            foreach (CTWContextTreeNode n in this.children.Values) {
                count += n.size();
            }
            return count;
        }
    }
}
