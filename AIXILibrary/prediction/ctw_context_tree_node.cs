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
        public double LogHalf = Math.Log(0.5);
        public Dictionary<int, CTWContextTreeNode> Children;
        public CTWContextTree Tree;
        public double LogKt=0.0;
        public double LogProbability = 0.0;//n: q: log of prob==0 -> prob ==1. Is this true?
        public int NumberOf0S=0;
        public int NumberOf1S=0;
        public CTWContextTreeNode(CTWContextTree tree){
            this.Children = new Dictionary<int, CTWContextTreeNode>();
            this.Tree = tree;
        }

        public void Print(int level = 0) {
            for (int i = 0; i < level; i++) {
                Console.Write("    ");
            }
            Console.Write("{0}, {1}, {2}-{3}   ", this.LogKt, this.LogProbability, this.SymbolCount(0), this.SymbolCount(1));
            foreach (KeyValuePair<int, CTWContextTreeNode> item in this.Children) {
                Console.Write("{0}: {1},  ", item.Key, item.Value);
            }
            Console.WriteLine();
            foreach (KeyValuePair<int, CTWContextTreeNode> item in this.Children)
            {
                item.Value.Print(level + 1);
            }
        }

        public bool IsLeaf() {
            return !this.Children.Any();
        }
        public int SymbolCount(int symbol) { 
            if (symbol == 0){
                return this.NumberOf0S;
            }
            else if (symbol == 1) {
                return this.NumberOf1S;
            }
            Debug.Assert(false, "Only symbols 0 and 1 are allowed");
            return -1;
        }

        public double LogKtMultiplier(int symbol) {
            //log of probability from KT-estimator:
            // log(Pr_kt(1 |0^a 1^b)) = log((b + 1/2)/(a + b + 1))
            double numerator = this.SymbolCount(symbol)+0.5;
            double denominator = this.Visits()+1;
            return Math.Log(numerator / denominator);
        }

        public int Visits() {
            return this.NumberOf0S + this.NumberOf1S;
        }

        public void SetSymbolCount(int symbol, int newValue){
            if (symbol==0){
                this.NumberOf0S=newValue;
            }
            else if (symbol == 1)
            {
                this.NumberOf1S = newValue;
            }
            else
            {
                Console.WriteLine("bad value: {0}", symbol);
                Debug.Assert(false, "trying to set symbol other than 0/1");
            }
        }

        public void Revert(int symbol) {
            int thisSymbolCount = this.SymbolCount(symbol);
            if (thisSymbolCount > 1)
            {
                this.SetSymbolCount(symbol, thisSymbolCount-1);
            }
            else {
                this.SetSymbolCount(symbol, 0);
            }

            if (this.Children.ContainsKey(symbol) && Children[symbol].Visits()==0) {    //note: this piece was not tested with rest of class
                this.Tree.TreeSize -= Children[symbol].Size();
                this.Children.Remove(symbol);    //low-todo: check if this truly free memory
            }

            this.LogKt -= this.LogKtMultiplier(symbol);
            this.UpdateLogProbability();
        }

        public void Update(int symbol) {
            this.LogKt += this.LogKtMultiplier(symbol);
            this.UpdateLogProbability();
            if (symbol == 0) {
                this.NumberOf0S += 1;
            }
            else if(symbol==1){
                this.NumberOf1S += 1;
            }
        }

        public void UpdateLogProbability() {
            if (this.IsLeaf())
            {
                this.LogProbability = this.LogKt;
            }
            else {
                double logChildProbability = 0;
                foreach (CTWContextTreeNode child in this.Children.Values) {
                    //note: this is not best way of doing summation of doubles. We will see if this will matter...
                    // (eg: python has math.fsum)
                    logChildProbability += child.LogProbability;
                }

                //for better numerical results (least chance of overflow)
                double a = Math.Max(this.LogKt, logChildProbability);
                double b = Math.Min(this.LogKt, logChildProbability);

                this.LogProbability = this.LogHalf + a + Utils.Log1P(Math.Exp(b - a));
            }
        }

        public int Size() {
            int count = 1;
            foreach (CTWContextTreeNode n in this.Children.Values) {
                count += n.Size();
            }
            return count;
        }
    }
}
