using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    public struct CtNode
    {
        public double LogKt;
        public double LogProbability;
        public int NumberOf0S;
        public int NumberOf1S;
        public int Child0;
        public int Child1;//child1 is 1, child0 is 0

    };
    public class CTWContextTreeFast : IModel
    {

        public int TreeSize;
        public List<int> Context;//speed: what about array of size depth?
        public int Depth;
        public List<int> History { get; set; }//speed: is this fully needed, cannot I just drop everything but end?
        //public CTWContextTreeNode root;
        public int RootI;

        public CtNode[] Nodes;

        public double[,] Multipliers;
        public Queue<int> FreeIndices;
        public int FirstFreeIndex;

        public int CacheMultipliersBellow = 200;

        public CTWContextTreeFast(int depth)
        {
            this.Nodes = new CtNode[256];
            this.FreeIndices = new Queue<int>();
            
            Context = new List<int>();
            //this.context.Capacity = this.depth * 2 + 1;
            
            Debug.Assert(depth >= 0);
            this.Depth = depth;
            this.History = new List<int> ();
            this.TreeSize = 1;

            this.Multipliers = new double[2*this.CacheMultipliersBellow, this.CacheMultipliersBellow];
            for (int sum = 0; sum < 2 * this.CacheMultipliersBellow; sum++)
            {
                for (int symbolCount = 0; symbolCount < this.CacheMultipliersBellow; symbolCount++)
                {
                    double numerator = 0.5+symbolCount;
                    double denominator = sum+1;
                    this.Multipliers[sum, symbolCount] = Math.Log(numerator / denominator);
                }
            }

            this.Clear();
        }

        public void print_node(int nodeIndex=-1, int level = 0) {
            if (nodeIndex == -1) {
                nodeIndex = this.RootI;
            }
            for (int i = 0; i < level; i++)
            {
                Console.Write("    ");
            }
            var node = this.Nodes[nodeIndex];
            Console.Write("{0}, {1}, {2}-{3}   ", node.LogKt, node.LogProbability, node.NumberOf0S, node.NumberOf1S);
            if (node.Child1 != -1) {
                Console.Write("{0}: {1},  ", "1:", node.Child1);
            }
            if (node.Child0 != -1)
            {
                Console.Write("{0}: {1},  ", "0:", node.Child0);
            }
            Console.WriteLine();
            if (node.Child1 != -1)
            {
                this.print_node(node.Child1, level + 1);
            }
            if (node.Child0 != -1)
            {
                this.print_node(node.Child0, level + 1);
            }
        }

        public void print_tree()
        {
            Console.Write("history ({0}): ", this.History.Count);
            foreach (var symbol in this.History)
            {
                Console.Write("{0},", symbol);
            }
            Console.WriteLine();
            Console.Write("context ({0}): ", this.Context.Count);
            foreach (var nodeI in this.Context)
            {
                Console.Write("{0},", nodeI);
            }
            Console.WriteLine();

            this.print_node(this.RootI);
        }

        public bool compare(CTWContextTree other) {
            if (this.TreeSize != other.TreeSize ||
                this.Depth != other.Depth)
            {
                return false;
            }

            for (int i = 0; i < this.Context.Count; i++) {
                if (!this.SameNode(this.Context[i], other.Context[i])) {
                    return false;
                }
            }
            for (int i=0; i<this.History.Count; i++){
                if (this.History[i] != other.History[i]){
                    return false;
                }
            }

            return this.compare(this.RootI, other.Root);
        }

        public bool SameNode(int meI, CTWContextTreeNode he) {
            var me = this.Nodes[meI];

            return Utils.FloatCompare(me.LogProbability, he.LogProbability) &&
                Utils.FloatCompare(me.LogKt, he.LogKt) &&
                Utils.FloatCompare(me.NumberOf0S, he.NumberOf0S) &&
                Utils.FloatCompare(me.NumberOf1S, he.NumberOf1S) &&
                (me.Child1 != -1) == he.Children.ContainsKey(1) &&
                (me.Child0 != -1) == he.Children.ContainsKey(0);
        }

        public bool compare(int meI, CTWContextTreeNode he) {
            var me = this.Nodes[meI];

            if (this.SameNode(meI, he))
            {
                bool result = true;

                if (me.Child1 != -1)
                {
                    result = result && this.compare(me.Child1, he.Children[1]);
                }
                if (me.Child0 != -1)
                {
                    result = result && this.compare(me.Child0, he.Children[0]);
                }
                return result;
            }
            return false;
        }

        public void Resize()
        {
            int oldSize = this.Nodes.Length;
            int newSize = oldSize * 2;    //speed: what about *4?
            CtNode[] newArray = new CtNode[newSize];
            int preserveLength = Math.Min(oldSize, newSize);
            if (preserveLength > 0){
                Array.Copy(this.Nodes, newArray, preserveLength);
            }
            this.Nodes = newArray;
        }


        public void Clear()
        {
            this.History = new List<int>();
            this.TreeSize = 1;
            this.Context = new List<int>();
            
            //this.nodes is still full, but is considered free and will be overwritten

            this.FreeIndices.Clear();
            FirstFreeIndex = 0;
            this.RootI = CreateNewNode();
        }
        public int CreateNewNode() {
            int index = this.GetFreeIndex();

            this.Nodes[index].LogProbability = 0.0;
            this.Nodes[index].LogKt = 0.0;
            this.Nodes[index].NumberOf0S = 0;
            this.Nodes[index].NumberOf1S = 0;
            this.Nodes[index].Child1 = -1;
            this.Nodes[index].Child0 = -1;

            return index;
        }

        public int GetFreeIndex() {
            //idea: use chache-oblivious placing (?)
            
            int index;
            if (this.FreeIndices.Count > 0)
            {
                index = this.FreeIndices.Dequeue();
            }
            else {
                if (this.Nodes.Length <= this.FirstFreeIndex) {
                    this.Resize();
                }
                index = this.FirstFreeIndex;
                this.FirstFreeIndex++;
            }
            return index;
        }

        public void FreeNode(int index) {//todo: should we change tree size here?
            this.FreeIndices.Enqueue(index);
        }

        public bool IsLeaf(int index)
        {
            return this.Nodes[index].Child1 == -1 &&
                this.Nodes[index].Child0 == -1;
        }

        public int SymbolCount(int index, int symbol)
        {
            if (symbol == 0)
            {
                return this.Nodes[index].NumberOf0S;
            }
            else if (symbol == 1)
            {
                return this.Nodes[index].NumberOf1S;
            }
            Debug.Assert(false, "Only symbols 0 and 1 are allowed");
            return -1;
        }

        public double LogKtMultiplier(int index, int symbol)
        {
            var node = this.Nodes[index];

            if (node.NumberOf0S < this.CacheMultipliersBellow && node.NumberOf1S < this.CacheMultipliersBellow)
            {
                if (symbol == 1)
                    return this.Multipliers[node.NumberOf0S+node.NumberOf1S, node.NumberOf1S];
                else {
                    return this.Multipliers[node.NumberOf0S + node.NumberOf1S, node.NumberOf0S];    
                }
            }
            
            double denominator = node.NumberOf0S + node.NumberOf1S + 1;
            double numerator;
            if (symbol == 1)
            {
                numerator = node.NumberOf1S + 0.5;
            }
            else
            {
                numerator = node.NumberOf0S + 0.5;
            }
            
            
            return Math.Log(numerator / denominator);
        }

        public int Visits(int index)
        {
            CtNode node = this.Nodes[index];
            return node.NumberOf0S + node.NumberOf1S;
        }

        public void SetSymbolCount(int index, int symbol, int newValue)
        {
            if (symbol == 0)
            {
                this.Nodes[index].NumberOf0S = newValue;
            }
            else if (symbol == 1)
            {
                this.Nodes[index].NumberOf1S = newValue;
            }
            else
            {
                Console.WriteLine("bad value: {0}", symbol);
                Debug.Assert(false, "trying to set symbol other than 0/1");
            }
        }

        public void revert_node(int index, int symbol)
        {
            int thisSymbolCount = this.SymbolCount(index, symbol);
            if (thisSymbolCount > 1)
            {
                this.SetSymbolCount(index, symbol, thisSymbolCount - 1);
            }
            else
            {
                this.SetSymbolCount(index, symbol, 0);
            }

            int lchild = this.Nodes[index].Child1;
            int rchild = this.Nodes[index].Child0;

            if (symbol==1 && lchild != -1) {
                if (this.FreeIfUnvisited(lchild))
                {
                    this.Nodes[index].Child1 = -1;
                }
            }
            if (symbol==0 && rchild != -1)
            {
                if (this.FreeIfUnvisited(rchild))
                {
                    this.Nodes[index].Child0 = -1;
                }
            }

            this.Nodes[index].LogKt -= this.LogKtMultiplier(index, symbol);
            this.UpdateLogProbability(index);
        }


        public bool FreeIfUnvisited(int index) {
            if (this.Visits(index)==0){
                this.FreeSubtree(index);
                return true;
            }
            else{
                return false;
            }
        }

        public void FreeSubtree(int index) {
            //note:this do not check if this node truly exists
            this.FreeIndices.Enqueue(index);
            this.TreeSize--;
            if (this.Nodes[index].Child1 != -1) {
                this.FreeSubtree(this.Nodes[index].Child1);
            }
            if (this.Nodes[index].Child0 != -1)
            {
                this.FreeSubtree(this.Nodes[index].Child0);
            }
        }


        public int subtree_size(int index) {
            int count = 1;
            if (this.Nodes[index].Child1!=-1){
                count += this.subtree_size(this.Nodes[index].Child1);
            }
            if (this.Nodes[index].Child0!=-1){
                count += this.subtree_size(this.Nodes[index].Child0);
            }
            return count;
        }

        public void update_node(int index, int symbol)
        {
            this.Nodes[index].LogKt += this.LogKtMultiplier(index, symbol);
            this.UpdateLogProbability(index);
            if (symbol == 0)
            {
                this.Nodes[index].NumberOf0S += 1;
            }
            else if (symbol == 1)
            {
                this.Nodes[index].NumberOf1S += 1;
            }
        }


        public void UpdateLogProbability(int index)
        {
            if (this.IsLeaf(index))
            {
                this.Nodes[index].LogProbability = this.Nodes[index].LogKt;
            }
            else
            {
                double logChildProbability = 0;
                int lchild = this.Nodes[index].Child1 ;
                if (lchild != -1) {
                    logChildProbability += this.Nodes[lchild].LogProbability;
                }
                int rchild = this.Nodes[index].Child0;
                if (rchild != -1)
                {
                    logChildProbability += this.Nodes[rchild].LogProbability;
                }

                //for better numerical results
                double a = Math.Max(this.Nodes[index].LogKt, logChildProbability);
                double b = Math.Min(this.Nodes[index].LogKt, logChildProbability);

                this.Nodes[index].LogProbability = Math.Log(0.5) + a + Utils.Log1P(Math.Exp(b - a));
                //todo: is it fast enough to compute Math.Log(0.5) every time. Joel has it cached.
            }
        }
        public int[] GenerateRandomSymbols(int symbolCount)
        {
            var symbol_list = this.GenerateRandomSymbolsAndUpdate(symbolCount);
            this.revert_tree(symbolCount);
            return symbol_list;
        }

        public int[] GenerateRandomSymbolsAndUpdate(int symbolCount)
        {
            int[] symbolList = new int[symbolCount];
            for (int i = 0; i < symbolCount; i++)
            {
                int symbol;
                var symbolsToPredict = new int[1];
                symbolsToPredict[0] = 1;

                if (Utils.Rnd.NextDouble() < this.Predict(symbolsToPredict))
                {
                    symbol = 1;
                }
                else
                {
                    symbol = 0;
                }
                symbolList[i] = symbol;

                var singletonSymbol = new int[1];
                singletonSymbol[0] = symbol;
                this.update_tree(singletonSymbol);
            }
            return symbolList;
        }

        public double Predict(int[] symbolList)
        {
            int symbolListLength = symbolList.Length;
            if (this.History.Count + symbolListLength <= this.Depth)
            {
                return Math.Pow(0.5, symbolListLength); //note: diff from pyaixi: removing if
            }

            double probHistory = this.Nodes[this.RootI].LogProbability;
            
            this.update_tree(symbolList);
            double probSequence = this.Nodes[this.RootI].LogProbability;
            this.revert_tree(symbolListLength);
            return Math.Exp(probSequence - probHistory);
        }

        public void update_tree(int[] symbolList)
        {
            foreach (int symbol in symbolList)
            {
                if (this.History.Count >= this.Depth)
                {
                    this.update_context();
                    for (int i = this.Depth - 1; i >= 0; i--)
                    {
                        int contextNodeI = this.Context[i];
                        this.update_node(contextNodeI, symbol);
                    }
                }
                this.update_tree_history(symbol);
            }
        }

        public void update_context()
        {
            Debug.Assert(this.History.Count >= this.Depth, "history is shorter than depth in update_context");
            this.Context = new List<int>();
            this.Context.Add(this.RootI);
            int nodeI = this.RootI;
            int updateDepth = 1;
            for (int i = History.Count-1; i>=0; i--)
            {
                int symbol = History[i];
                var node = this.Nodes[nodeI];
                if (symbol == 1 && node.Child1 != -1) {
                    nodeI = node.Child1;
                }
                else if (symbol == 0 && node.Child0 != -1) {
                    nodeI = node.Child0;
                }
                else
                {
                    int newNodeI = this.CreateNewNode();
                    if (symbol == 1) {
                        this.Nodes[nodeI].Child1 = newNodeI;
                    }
                    else if (symbol == 0)
                    {
                        this.Nodes[nodeI].Child0 = newNodeI;
                    }
                    else {
                        Debug.Assert(false, "invalid symbol");
                    }
                    nodeI = newNodeI;
                    this.TreeSize += 1;
                }
                node = this.Nodes[nodeI];

                this.Context.Add(nodeI);
                updateDepth += 1;
                if (updateDepth > this.Depth)
                {
                    break;
                }
            }
        }

        public void revert_tree(int symbolCount = 1)
        {
            for (int i = 0; i < symbolCount; i++)
            {
                if (this.History.Count == 0)
                {
                    return;
                }
                int symbol = this.History.Last();
                this.History.RemoveAt(this.History.Count - 1);

                if (this.History.Count >= this.Depth)
                {
                    this.update_context();
                    for (int j = this.Depth - 1; j >= 0; j--)
                    {
                        int nodeI = this.Context[j];
                        this.revert_node(nodeI, symbol);
                    }
                }
            }
        }

        public void revert_tree_history(int symbolCount)
        {
            Debug.Assert(symbolCount >= 0);
            int historyLength = this.History.Count;
            Debug.Assert(historyLength >= symbolCount);
            //int newSize = historyLength - symbolCount;
            //this.history = this.history.GetRange(0, new_size);
            this.History.RemoveRange(historyLength-symbolCount, symbolCount);
        }


        public void update_tree_history(int symbol)
        {
            this.History.Add(symbol);
        }
        public void update_tree_history(int[] symbolList)
        {
            foreach (int symbol in symbolList)
            {
                this.update_tree_history(symbol);
            }
        }

        public int get_model_size()
        {
            return this.TreeSize;
        }
    }
}
