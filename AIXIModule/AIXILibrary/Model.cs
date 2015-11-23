using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    public interface IModel
    {
        void Clear();
        void update_tree(int[] symbolList);
        void update_tree_history(int[] symbolList);
        void update_tree_history(int symbol);
        
        void revert_tree(int symbolCount = 1);
        void revert_tree_history(int symbolCount);

        double Predict(int[] symbolList);
        int[] GenerateRandomSymbolsAndUpdate(int symbolCount);
        int[] GenerateRandomSymbols(int symbolCount);


        int get_model_size();

        List<int> History {get; set;}



    }
}
