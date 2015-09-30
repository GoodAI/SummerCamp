using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    public class CtwContextTreeUndo
    {
        public int Age;
        public double TotalReward;
        public int HistorySize;
        public int LastUpdate;

        public CtwContextTreeUndo(MC_AIXI_CTW agent)
        {
            this.Age = agent.Age;
            this.TotalReward = agent.TotalReward;
            this.HistorySize = agent.history_size();
            this.LastUpdate = agent.LastUpdate;
        }
    }
}
