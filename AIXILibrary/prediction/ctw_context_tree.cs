using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    public class CTWContextTree : IModel{
        public int TreeSize;
        public List<CTWContextTreeNode> Context;
        public int Depth;//refact: is this final/current depth?
        public List<int> History { get; set; }
        public CTWContextTreeNode Root;
        public CTWContextTree(int depth)
        {
            Context = new List<CTWContextTreeNode>();
            Debug.Assert(depth >= 0);
            this.Depth = depth;
            this.History = new List<int> ();
            this.Root = new CTWContextTreeNode(this);   //refact: most of this is in clear()
            this.TreeSize = 1;
            this.Clear();
        }

        public void Clear() { 
            this.History = new List<int>();
            this.Root.Tree = null; //TODO: is this working - will it free memory? Test!
            //missing?: del self.root
            this.Root = new CTWContextTreeNode(this);
            this.TreeSize = 1;
            this.Context = new List<CTWContextTreeNode>();
        }

        public void print_tree() {
            Console.Write("history ({0}): ", this.History.Count);
            foreach (var symbol in this.History) {
                Console.Write("{0},", symbol);
            }
            Console.WriteLine();

            Console.Write("context ({0}): ", this.Context.Count);
            foreach (var node in this.Context)
            {
                Console.Write("{0},", node.LogProbability);
            }
            Console.WriteLine();

            this.Root.Print();
        }

        public void GenerateRandomSymbols(int symbolCount) {
            //TODO, I do not understand this one - it should not be needed
            throw new NotImplementedException();
        }

        public int[] GenerateRandomSymbolsAndUpdate(int symbolCount) {
            int[] symbolList = new int[symbolCount];
            for (int i = 0; i < symbolCount; i++) {
                int symbol;
                var symbolsToPredict = new int[1];
                symbolsToPredict[0]=1;


                if (Utils.Rnd.NextDouble() < this.Predict(symbolsToPredict)){
                    symbol=1;
                }
                else{
                    symbol = 0;
                }
                symbolList[i] = symbol;

                var singletonSymbol = new int[1];
                singletonSymbol[0]=symbol;
                this.update_tree(singletonSymbol);
            }
            return symbolList;
        }
        public double Predict(int[] symbolList) {
            int symbolListLength = symbolList.Length;
            if (this.History.Count + symbolListLength <= this.Depth) {
                return Math.Pow(0.5, symbolListLength); //note: diff from pyaixi: removing if
            }

            double probHistory = this.Root.LogProbability;
            this.update_tree(symbolList);
            double probSequence = this.Root.LogProbability;
            this.revert_tree(symbolListLength);
            return Math.Exp(probSequence - probHistory);
        }

        public void update_tree(int[] symbolList) {
            foreach (int symbol in symbolList) {
                if (this.History.Count >= this.Depth) {
                    this.update_context();
                    for (int i = this.Depth-1; i >=0; i--)
                    {
                        CTWContextTreeNode contextNode = this.Context[i];
                        contextNode.Update(symbol);
                    }
                }
                this.update_tree_history(symbol);
            }
        }


        public void update_context()
        {
            Debug.Assert(this.History.Count >= this.Depth, "history is shorter than depth in update_context");
            this.Context = new List<CTWContextTreeNode>();
            this.Context.Add(this.Root);
            CTWContextTreeNode node = this.Root;
            int updateDepth = 1;
            IEnumerable<int> historyIe = History;
            foreach (int symbol in historyIe.Reverse())
            { //TODO: will this save reversed history to this.history?
                if (node.Children.ContainsKey(symbol))
                {
                    node = node.Children[symbol];
                }
                else
                {
                    CTWContextTreeNode newNode = new CTWContextTreeNode(this);
                    node.Children[symbol] = newNode;
                    this.TreeSize += 1;
                    node = newNode;
                }
                this.Context.Add(node);
                updateDepth += 1;
                if (updateDepth > this.Depth)
                {
                    break;
                }
            }
        }


        public void revert_tree(int symbolCount = 1)
        {
            for (int i = 0; i < symbolCount; i++) {
                if (this.History.Count == 0) {
                    return;
                }
                int symbol = this.History.Last();
                this.History.RemoveAt(this.History.Count-1);

                if (this.History.Count >= this.Depth) {
                    this.update_context();
                    //refact: We do not need to create variable nodes at all
                    for (int j = this.Depth - 1; j >= 0; j--) {
                        CTWContextTreeNode node = this.Context[j];
                        node.Revert(symbol);
                    }
                }
            }
        }

        public void revert_tree_history(int symbolCount) {
            Debug.Assert(symbolCount>=0);
            int historyLength = this.History.Count;
            Debug.Assert(historyLength >= symbolCount);
            int newSize = historyLength - symbolCount;
            this.History = this.History.GetRange(0,newSize);
        }

        public void update_tree_history(int symbol) {
            this.History.Add(symbol);
        }
        public void update_tree_history(int[] symbolList) {
            foreach (int symbol in symbolList) {
                this.update_tree_history(symbol);
            }
        }

        public int get_model_size() {
            return this.TreeSize;
        }

    }
}
